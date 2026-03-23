"""
Preprocessing Script: SOTA Minimal Regex (Phase 1)
Extracts raw 'Content' from HDFS_full.log_structured.csv, applies Minimal Regex 
to mask <IP> and <BLK>, avoiding Drain's aggressive parsing but preventing vocabulary explosion.
Aligns exactly with the session subsets defined in hdfs_train_5k.csv and hdfs_test_subset.csv.
"""
import polars as pl
import re
import time
from pathlib import Path

from config import (
    LAB_DATA_DIR, DATA_DIR, SESSION_ID_COL, LABEL_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Paths
HDFS_RAW_STRUCTURED = DATA_DIR / "HDFS" / "HDFS_full.log_structured.csv"
TRAIN_SUBSET = LAB_DATA_DIR / "hdfs_train_5k.csv"
TEST_SUBSET = LAB_DATA_DIR / "hdfs_test_subset.csv"

OUT_TRAIN = LAB_DATA_DIR / "hdfs_train_5k_regex.csv"
OUT_TEST = LAB_DATA_DIR / "hdfs_test_subset_regex.csv"

def minimal_regex_filter(text):
    """
    State-of-the-Art Minimal Regex (LogLLM inspired)
    Replaces purely dynamic noise without destroying semantic structure.
    """
    # 1. Mask HDFS Block IDs
    text = re.sub(r'blk_-?\d+', '<BLK>', text)
    # 2. Mask IPs with Ports (e.g., /10.251.43.115:50010 or 10.251.43.115:50010)
    text = re.sub(r'/?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', '<IP>', text)
    # 3. Mask solitary IPs
    text = re.sub(r'/?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', text)
    # 4. Mask numbers that are likely sizes/offsets to reduce vocab (optional but helps heavily)
    # text = re.sub(r'\b\d+\b', '<NUM>', text) # We will only do this if vocab explodes, keeping it minimal for now.
    
    return text

def process_and_align():
    logger.info("="*60)
    logger.info("SOTA PREPROCESSING: Minimal Regex Generator")
    logger.info("="*60)
    
    # 1. Load target session IDs
    logger.info("Loading target session subsets...")
    train_ids = pl.read_csv(str(TRAIN_SUBSET))[SESSION_ID_COL].unique().to_list()
    test_ids = pl.read_csv(str(TEST_SUBSET))[SESSION_ID_COL].unique().to_list()
    target_ids = set(train_ids + test_ids)
    
    logger.info(f"Target Train Sessions: {len(train_ids):,}")
    logger.info(f"Target Test Sessions:  {len(test_ids):,}")
    logger.info(f"Total Unique Targets:  {len(target_ids):,}")
    
    # We also need labels to attach to the final df
    labels_df = pl.concat([
        pl.read_csv(str(TRAIN_SUBSET)).select([SESSION_ID_COL, LABEL_COL]).unique(),
        pl.read_csv(str(TEST_SUBSET)).select([SESSION_ID_COL, LABEL_COL]).unique()
    ]).unique()
    
    label_map = dict(zip(labels_df[SESSION_ID_COL].to_list(), labels_df[LABEL_COL].to_list()))
    
    # 2. Process Raw Structured Log
    # HDFS_full.log_structured.csv is large (~1.7GB). We can read it in chunks or use Polars lazy frame.
    logger.info(f"Scanning raw log file: {HDFS_RAW_STRUCTURED}...")
    t0 = time.time()
    
    # Since we need to extract blk_id via regex to match session_id, we will do it in Polars memory
    # We only need Content
    df_raw = pl.read_csv(str(HDFS_RAW_STRUCTURED), columns=["Content"], infer_schema_length=0)
    
    logger.info(f"Loaded {len(df_raw):,} raw logs in {time.time() - t0:.1f}s")
    
    logger.info("Extracting Session IDs (blk_)....")
    # Extract blk_id
    df_raw = df_raw.with_columns(
        pl.col("Content")
          .str.extract(r'(blk_-?\d+)', 1)
          .alias(SESSION_ID_COL)
    )
    
    # Filter only target sessions
    logger.info("Filtering to match target lab subsets...")
    df_filtered = df_raw.filter(pl.col(SESSION_ID_COL).is_in(list(target_ids)))
    logger.info(f"Retained {len(df_filtered):,} logs belonging to target sessions.")
    
    # 3. Apply Minimal Regex
    logger.info("Applying Minimal Regex Filter to raw Content...")
    t1 = time.time()
    
    # Polars string replacement
    df_filtered = df_filtered.with_columns(
        pl.col("Content")
          .str.replace_all(r'blk_-?\d+', '<BLK>')
          .str.replace_all(r'/?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', '<IP>')
          .str.replace_all(r'/?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>')
          # Masking isolated numeric values (sizes, ports, ms) as <NUM> to emulate LogFiT robustness
          .str.replace_all(r'\b\d+\b', '<NUM>')
          .alias("EventTemplate_Regex") # We call it template_regex so it connects seamlessly to the pipeline
    )
    
    logger.info(f"Regex applied in {time.time() - t1:.1f}s")
    
    # We don't have absolute timestamps in Content anymore, we will just preserve sequential order
    # Add a mock abstract timestamp to satisfy pipeline ordering requirements
    df_filtered = df_filtered.with_row_count("seq_id")
    df_filtered = df_filtered.with_columns(
        pl.col("seq_id").alias("timestamp")
    )
    
    # Mapear labels
    logger.info("Mapping anomaly labels...")
    def get_label(s_id):
        return label_map.get(s_id, 0)
        
    # We have to do this via map_elements or join. Join is faster.
    df_labels = pl.DataFrame({
        SESSION_ID_COL: list(label_map.keys()),
        LABEL_COL: list(label_map.values())
    })
    
    df_final = df_filtered.join(df_labels, on=SESSION_ID_COL, how="inner")
    
    # We rename 'EventTemplate_Regex' to 'EventTemplate' so `dataset.py` natively supports it
    df_final = df_final.select([
        SESSION_ID_COL,
        "timestamp",
        pl.col("EventTemplate_Regex").alias("EventTemplate"),
        LABEL_COL
    ])
    
    # 4. Split and Save
    df_train = df_final.filter(pl.col(SESSION_ID_COL).is_in(train_ids))
    df_test = df_final.filter(pl.col(SESSION_ID_COL).is_in(test_ids))
    
    logger.info(f"Saving Regex Train Subset: {len(df_train):,} logs -> {OUT_TRAIN}")
    df_train.write_csv(str(OUT_TRAIN))
    
    logger.info(f"Saving Regex Test Subset: {len(df_test):,} logs -> {OUT_TEST}")
    df_test.write_csv(str(OUT_TEST))
    
    logger.info("="*60)
    logger.info("✅ SOTA Preprocessing Complete!")
    logger.info("="*60)

if __name__ == '__main__':
    process_and_align()
