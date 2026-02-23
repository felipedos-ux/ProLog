"""
Dataset preparation for LogGPT training.
Identical logic to OpenStack: groups by session, concatenates templates, tokenizes, chunks.
"""

import polars as pl
from pathlib import Path
from typing import Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer

from config import DATA_FILE, SESSION_ID_COL, TIMESTAMP_COL, TEMPLATE_COL, LABEL_COL

DATA_PATH = DATA_FILE


def load_data(data_path: Path = DATA_PATH) -> pl.DataFrame:
    """Loads and filters data for LLM training (Normal only)"""
    print(f"📦 Loading data from {data_path}...")
    df = pl.read_csv(str(data_path), infer_schema_length=10000)
    
    # Filter Normal logs for training and ensure template exists
    normal_df = df.filter(
        (pl.col(LABEL_COL) == 0) & 
        (pl.col(TEMPLATE_COL).is_not_null())
    )
    print(f"✅ Loaded {len(normal_df)} normal logs for training.")
    return normal_df


def prepare_llm_dataset(
    tokenizer: PreTrainedTokenizer, 
    block_size: int = 128,
    data_path: Optional[Path] = None
):
    """
    Prepares a Hugging Face Dataset for Causal LM.
    
    Strategy (same as OpenStack):
    1. Group logs by session (session_id).
    2. Sort by timestamp within each session.
    3. Concatenate EventTemplate with newline separator.
    4. Tokenize the full session text.
    5. Chunk into block_size for training.
    """
    path = data_path or DATA_PATH
    df = load_data(path)
    
    # Group by session and concat templates
    print(f"🔄 Grouping logs by session ({SESSION_ID_COL})...")
    
    sessions = (
        df.sort(TIMESTAMP_COL)
        .group_by(SESSION_ID_COL)
        .agg(pl.col(TEMPLATE_COL))
        .select(TEMPLATE_COL)
    )
    
    # Convert to list of strings
    text_sessions = []
    
    rows = sessions.rows()
    for row in rows:
        session_text = " \n ".join(str(t) for t in row[0])
        text_sessions.append(session_text)
        
    print(f"📄 Created {len(text_sessions)} session documents.")
    
    # Create HF Dataset
    dataset = Dataset.from_dict({"text": text_sessions})
    
    # Tokenization Function
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    print("🔤 Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=12,  # Use all 12 threads of Ryzen 3600
        remove_columns=["text"]
    )
    
    # Grouping Function (Chunking)
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print(f"✂️ Chunking into blocks of {block_size} tokens...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=2000,  # Larger batch size for faster processing
        num_proc=12,  # Use all 12 threads
    )
    
    return lm_datasets


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    ds = prepare_llm_dataset(tokenizer)
    print(ds)
