"""
HDFS Preprocessor: Converts raw HDFS data to OpenStack-compatible format.

Input:  Event_traces.csv + HDFS_full.log_templates.csv + anomaly_label.csv
Output: HDFS_data_processed.csv with columns:
        session_id | timestamp | EventTemplate | anom_label
"""

import polars as pl
import ast
from pathlib import Path
from datetime import datetime, timedelta

from config import (
    EVENT_TRACES_FILE, TEMPLATES_FILE, LABELS_FILE, DATA_FILE,
    SESSION_ID_COL, TIMESTAMP_COL, TEMPLATE_COL, LABEL_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_list_string(s: str) -> list:
    """Parse a string like '[E5,E22,E5]' into a list."""
    if not s or s == '[]':
        return []
    # Remove brackets and split
    s = s.strip('[]')
    return [x.strip().strip("'\"") for x in s.split(',')]


def main():
    logger.info("🔄 Starting HDFS Preprocessing...")
    
    # 1. Load Event_traces.csv
    logger.info(f"Loading Event Traces from {EVENT_TRACES_FILE}...")
    traces = pl.read_csv(str(EVENT_TRACES_FILE))
    logger.info(f"  Loaded {len(traces)} sessions")
    logger.info(f"  Columns: {traces.columns}")
    
    # 2. Load Templates (EventId -> EventTemplate mapping)
    logger.info(f"Loading Templates from {TEMPLATES_FILE}...")
    templates = pl.read_csv(str(TEMPLATES_FILE))
    eid_to_template = dict(zip(
        templates["EventId"].to_list(),
        templates["EventTemplate"].to_list()
    ))
    logger.info(f"  Loaded {len(eid_to_template)} templates")
    
    # 3. Load Anomaly Labels
    logger.info(f"Loading Labels from {LABELS_FILE}...")
    labels = pl.read_csv(str(LABELS_FILE))
    label_map = dict(zip(
        labels["BlockId"].to_list(),
        labels["Label"].to_list()
    ))
    logger.info(f"  Loaded {len(label_map)} labels")
    logger.info(f"  Anomalies: {sum(1 for v in label_map.values() if v == 'Anomaly')}")
    
    # 4. Expand traces into individual log lines
    # Each row in Event_traces has Features=[E5,E22,...] and TimeInterval=[0.0, 1.0, ...]
    # We create one row per log event with reconstructed timestamps
    
    logger.info("Expanding sessions into individual log lines...")
    
    all_rows = []
    base_time = datetime(2008, 11, 9, 0, 0, 0)  # HDFS logs start around this date
    
    for row in traces.iter_rows(named=True):
        block_id = row["BlockId"]
        features_str = row["Features"]
        time_str = row["TimeInterval"]
        
        # Parse lists
        event_ids = parse_list_string(features_str)
        
        # Parse time intervals
        try:
            time_intervals = parse_list_string(time_str)
            time_intervals = [float(t) for t in time_intervals]
        except (ValueError, TypeError):
            time_intervals = [0.0] * len(event_ids)
        
        # Get label
        label_str = label_map.get(block_id, row.get("Label", "Normal"))
        anom_label = 1 if label_str in ("Anomaly", "Fail") else 0
        
        # Build rows: one per event
        cumulative_seconds = 0.0
        for i, eid in enumerate(event_ids):
            if i < len(time_intervals):
                cumulative_seconds += time_intervals[i]
            
            template = eid_to_template.get(eid, f"Unknown_{eid}")
            ts = base_time + timedelta(seconds=cumulative_seconds)
            
            all_rows.append({
                SESSION_ID_COL: block_id,
                TIMESTAMP_COL: ts.isoformat(),
                TEMPLATE_COL: template,
                LABEL_COL: anom_label
            })
        
        # Advance base_time for next session to avoid timestamp collisions
        base_time += timedelta(seconds=cumulative_seconds + 10)
    
    # 5. Create DataFrame and save
    logger.info(f"Creating DataFrame with {len(all_rows)} rows...")
    df = pl.DataFrame(all_rows)
    
    # Stats
    n_sessions = df[SESSION_ID_COL].n_unique()
    n_normal = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].n_unique()
    n_anomaly = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].n_unique()
    
    logger.info(f"  Total Sessions: {n_sessions}")
    logger.info(f"  Normal: {n_normal}, Anomaly: {n_anomaly}")
    logger.info(f"  Total Log Lines: {len(df)}")
    
    # Save
    logger.info(f"Saving to {DATA_FILE}...")
    df.write_csv(str(DATA_FILE))
    
    logger.info(f"✅ Done! Output: {DATA_FILE}")
    logger.info(f"   File size: {DATA_FILE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
