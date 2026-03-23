"""
Data Sampler: Creates reduced subsets from the full HDFS dataset.
Generates train/test CSVs small enough for fast experimentation (~5-10 min each).
"""
import polars as pl
import time
from pathlib import Path

from config import (
    SOURCE_DATA_FILE, LAB_DATA_DIR,
    SAMPLE_TRAIN_SESSIONS, SAMPLE_TEST_NORMAL, SAMPLE_SEED,
    INFER_SCHEMA_LENGTH,
    SESSION_ID_COL, TEMPLATE_COL, LABEL_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


def create_subsets():
    """
    Creates train and test subsets from the full HDFS dataset.
    
    Strategy:
    - Train: SAMPLE_TRAIN_SESSIONS random normal sessions
    - Test: SAMPLE_TEST_NORMAL random normal sessions + ALL anomalous sessions
    - Ensures no overlap between train and test normal sessions
    """
    logger.info("=" * 60)
    logger.info("DATA SAMPLER: Creating Lab Subsets")
    logger.info("=" * 60)
    
    # Load full dataset
    logger.info(f"Loading full dataset from {SOURCE_DATA_FILE}...")
    t0 = time.time()
    df = pl.read_csv(str(SOURCE_DATA_FILE), infer_schema_length=INFER_SCHEMA_LENGTH)
    load_time = time.time() - t0
    logger.info(f"Loaded {len(df):,} rows in {load_time:.1f}s")
    
    # Get unique session IDs by label
    normal_ids = df.filter(pl.col(LABEL_COL) == 0)[SESSION_ID_COL].unique().to_list()
    anom_ids = df.filter(pl.col(LABEL_COL) == 1)[SESSION_ID_COL].unique().to_list()
    
    logger.info(f"Total sessions: {len(normal_ids) + len(anom_ids):,}")
    logger.info(f"  Normal: {len(normal_ids):,}")
    logger.info(f"  Anomalous: {len(anom_ids):,}")
    
    # Sample normal sessions for train and test (no overlap)
    import random
    random.seed(SAMPLE_SEED)
    shuffled_normal = normal_ids.copy()
    random.shuffle(shuffled_normal)
    
    train_ids = shuffled_normal[:SAMPLE_TRAIN_SESSIONS]
    test_normal_ids = shuffled_normal[SAMPLE_TRAIN_SESSIONS:SAMPLE_TRAIN_SESSIONS + SAMPLE_TEST_NORMAL]
    
    logger.info(f"\nSampled subsets:")
    logger.info(f"  Train (normal only): {len(train_ids):,} sessions")
    logger.info(f"  Test normal: {len(test_normal_ids):,} sessions")
    logger.info(f"  Test anomalous: {len(anom_ids):,} sessions (ALL)")
    
    # Extract train and test DataFrames
    train_df = df.filter(pl.col(SESSION_ID_COL).is_in(train_ids))
    test_normal_df = df.filter(pl.col(SESSION_ID_COL).is_in(test_normal_ids))
    test_anom_df = df.filter(pl.col(SESSION_ID_COL).is_in(anom_ids))
    test_df = pl.concat([test_normal_df, test_anom_df])
    
    # Stats
    train_templates = train_df[TEMPLATE_COL].n_unique()
    test_templates = test_df[TEMPLATE_COL].n_unique()
    
    logger.info(f"\nData sizes:")
    logger.info(f"  Train: {len(train_df):,} rows ({train_templates} unique templates)")
    logger.info(f"  Test:  {len(test_df):,} rows ({test_templates} unique templates)")
    
    # Save subsets
    train_path = LAB_DATA_DIR / "hdfs_train_5k.csv"
    test_path = LAB_DATA_DIR / "hdfs_test_subset.csv"
    
    train_df.write_csv(str(train_path))
    test_df.write_csv(str(test_path))
    
    train_size_mb = train_path.stat().st_size / (1024 * 1024)
    test_size_mb = test_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"\nSaved:")
    logger.info(f"  {train_path} ({train_size_mb:.1f} MB)")
    logger.info(f"  {test_path} ({test_size_mb:.1f} MB)")
    
    # Save stats for reference
    stats = {
        'source_file': str(SOURCE_DATA_FILE),
        'total_rows': len(df),
        'total_sessions': len(normal_ids) + len(anom_ids),
        'train_sessions': len(train_ids),
        'test_normal_sessions': len(test_normal_ids),
        'test_anom_sessions': len(anom_ids),
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_unique_templates': train_templates,
        'test_unique_templates': test_templates,
        'seed': SAMPLE_SEED,
    }
    
    import json
    stats_path = LAB_DATA_DIR / "sampling_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  {stats_path}")
    
    logger.info("=" * 60)
    logger.info("✅ Data sampling complete!")
    logger.info("=" * 60)
    
    return stats


if __name__ == '__main__':
    stats = create_subsets()
