
import polars as pl
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from config import DATA_FILE, LOG_COLUMN, BLOCK_SIZE

DATA_PATH = DATA_FILE


class LogSessionDataset(TorchDataset):
    """
    Dataset that treats each session as a separate training example.
    Identical to HDFS HDFSDataset: tokenizes session strings and stores as tensors.
    
    Each session is a space-separated sequence of EventIds: "E1 E2 E5 E1 E3"
    """
    
    def __init__(self, data_list, tokenizer, block_size):
        self.examples = []
        
        print(f"  Tokenizing {len(data_list)} sessions...")
        
        MAX_SEQ_LEN = 1024  # GPT2 max position embeddings
        
        batch_size = 1000
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            encodings = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_SEQ_LEN,  # Cap at GPT2 max, not BLOCK_SIZE
                padding=False,
                return_attention_mask=False
            )
            
            for input_ids in encodings['input_ids']:
                self.examples.append(torch.tensor(input_ids, dtype=torch.long))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        return self.examples[item]


def collate_fn(batch):
    """Dynamic padding with EOS token (same as HDFS)."""
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), 50256, dtype=torch.long)
    
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    
    return padded


def load_openstack_data(data_path: Path = DATA_PATH) -> pl.DataFrame:
    """Loads OpenStack data."""
    print(f"📦 Loading data from {data_path}...")
    df = pl.read_csv(str(data_path), infer_schema_length=10000)
    return df


def prepare_session_strings(df: pl.DataFrame, label_filter: Optional[int] = None) -> pl.DataFrame:
    """
    Groups logs by test_id and creates space-separated EventId strings.
    Identical to HDFS dataset_hdfs.py line 209:
        .with_columns(pl.col("EventId").list.join(" ").alias("EventTemplate"))
    """
    filtered = df
    if label_filter is not None:
        filtered = df.filter(pl.col("anom_label") == label_filter)
    
    filtered = filtered.filter(pl.col(LOG_COLUMN).is_not_null())
    
    sessions = (
        filtered.sort("timestamp")
        .group_by("test_id")
        .agg([
            pl.col(LOG_COLUMN),
            pl.col("anom_label").max().alias("label")  # max() not first(): any anom log = anom session
        ])
    )
    
    # Join EventIds with space (like HDFS)
    sessions = sessions.with_columns(
        pl.col(LOG_COLUMN).list.join(" ").alias("EventTemplate")
    )
    
    return sessions


if __name__ == "__main__":
    df = load_openstack_data()
    sessions = prepare_session_strings(df, label_filter=0)
    
    print(f"\nNormal sessions: {len(sessions)}")
    print(f"Sample EventTemplate: {sessions['EventTemplate'][0][:200]}...")
    
    # Check token count
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sample = sessions["EventTemplate"][0]
    tokens = tokenizer.encode(sample)
    print(f"\nSample session: {len(sample.split())} EventIds -> {len(tokens)} GPT2 tokens")
