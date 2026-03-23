"""
Dataset preparation for HDFS Lab experiments.
Adds deduplication support (SiaLog-style) and max_sessions limiting.
"""
import polars as pl
from pathlib import Path
from typing import Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer

from config import (
    LAB_DATA_DIR, SESSION_ID_COL, TIMESTAMP_COL, TEMPLATE_COL, LABEL_COL
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

TRAIN_FILE = LAB_DATA_DIR / "hdfs_train_5k.csv"


def load_train_data(data_path: Path = TRAIN_FILE) -> pl.DataFrame:
    """Loads training data (Normal sessions only)."""
    logger.info(f"Loading train data from {data_path}...")
    df = pl.read_csv(str(data_path), infer_schema_length=10000)
    
    # Filter Normal only and ensure template exists
    normal_df = df.filter(
        (pl.col(LABEL_COL) == 0) &
        (pl.col(TEMPLATE_COL).is_not_null())
    )
    logger.info(f"Loaded {len(normal_df):,} normal log rows for training.")
    return normal_df


def deduplicate_sessions(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    Remove duplicate sessions (SiaLog-style).
    Sessions with identical template sequences are considered duplicates.
    
    Returns:
        df_deduped: DataFrame with only unique sessions
        stats: dict with deduplication statistics
    """
    logger.info("Deduplicating sessions...")
    
    # Group by session, create sequence string
    sessions = (
        df.sort(TIMESTAMP_COL)
        .group_by(SESSION_ID_COL)
        .agg([
            pl.col(TEMPLATE_COL).alias('templates'),
            pl.col(LABEL_COL).first().alias('label_first')
        ])
    )
    
    # Create hash from template sequence
    sessions = sessions.with_columns(
        pl.col('templates').list.join(",").alias('seq_hash')
    )
    
    original_count = len(sessions)
    
    # Keep unique sequences only
    sessions_unique = sessions.unique(subset=['seq_hash'])
    final_count = len(sessions_unique)
    
    reduction_pct = 100 * (1 - final_count / original_count) if original_count > 0 else 0
    
    stats = {
        'original_sessions': original_count,
        'unique_sessions': final_count,
        'removed': original_count - final_count,
        'reduction_pct': round(reduction_pct, 1),
    }
    
    logger.info(f"  Original: {original_count:,} sessions")
    logger.info(f"  Unique:   {final_count:,} sessions")
    logger.info(f"  Removed:  {stats['removed']:,} ({reduction_pct:.1f}%)")
    
    # Get the unique session IDs
    unique_ids = sessions_unique[SESSION_ID_COL].to_list()
    
    # Filter original df to keep only unique sessions
    df_deduped = df.filter(pl.col(SESSION_ID_COL).is_in(unique_ids))
    
    return df_deduped, stats


def prepare_llm_dataset(
    tokenizer: PreTrainedTokenizer,
    block_size: int = 128,
    data_path: Optional[Path] = None,
    deduplicate: bool = False,
    max_sessions: Optional[int] = None,
    is_regex: bool = False,
):
    """
    Prepares a HuggingFace Dataset for Causal LM.
    
    Args:
        tokenizer: HF tokenizer
        block_size: token sequence length per chunk
        data_path: path to CSV, defaults to lab train file
        deduplicate: if True, removes duplicate sessions (SiaLog-style)
        max_sessions: if set, limits number of sessions used
        is_regex: if True, loads the parser-free regex dataset
    
    Returns:
        lm_datasets: HF Dataset ready for training
        metadata: dict with stats about the dataset
    """
    if is_regex and data_path is None:
        path = LAB_DATA_DIR / "hdfs_train_5k_regex.csv"
    else:
        path = data_path or TRAIN_FILE
        
    df = load_train_data(path)
    
    metadata = {'total_rows_loaded': len(df)}
    
    # Deduplication (Experiment C, F)
    dedupe_stats = None
    if deduplicate:
        df, dedupe_stats = deduplicate_sessions(df)
        metadata['deduplication'] = dedupe_stats
    
    # Group by session and concat templates
    logger.info(f"Grouping logs by session ({SESSION_ID_COL})...")
    
    sessions = (
        df.sort(TIMESTAMP_COL)
        .group_by(SESSION_ID_COL)
        .agg(pl.col(TEMPLATE_COL))
    )
    
    # Limit sessions if requested
    if max_sessions and len(sessions) > max_sessions:
        sessions = sessions.sample(n=max_sessions, seed=42)
        logger.info(f"Limited to {max_sessions} sessions")
    
    # Count unique templates
    all_templates = df[TEMPLATE_COL].unique()
    metadata['unique_templates'] = len(all_templates)
    metadata['sessions_used'] = len(sessions)
    
    logger.info(f"Unique templates: {metadata['unique_templates']}")
    logger.info(f"Sessions to train: {len(sessions)}")
    
    # Convert to list of strings
    text_sessions = []
    rows = sessions.select(TEMPLATE_COL).rows()
    for row in rows:
        session_text = " \n ".join(str(t) for t in row[0])
        text_sessions.append(session_text)
    
    logger.info(f"Created {len(text_sessions)} session documents.")
    
    # Create HF Dataset
    dataset = Dataset.from_dict({"text": text_sessions})
    
    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    logger.info("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )
    
    # Chunking
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    logger.info(f"Chunking into blocks of {block_size} tokens...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=2000,
        num_proc=4,
    )
    
    metadata['total_chunks'] = len(lm_datasets)
    logger.info(f"Total chunks: {metadata['total_chunks']}")
    
    return lm_datasets, metadata
