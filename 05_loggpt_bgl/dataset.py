
import polars as pl
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer

from config import DATA_FILE, SESSION_MAX_LOGS, SESSION_ID_COL, LABEL_COL

# Reuse config from stable if possible, or define new constants
DATA_PATH = DATA_FILE # From config.py

def load_bgl_data(data_path: Path = DATA_PATH) -> pl.DataFrame:
    """Loads and filters BGL data for LLM training (Normal only)"""
    print(f"📦 Loading data from {data_path}...")
    df = pl.read_csv(str(data_path), infer_schema_length=10000)
    
    # Filter Normal logs for training and ensure template exists
    # BGL usa 'label' (não 'anom_label')
    normal_df = df.filter(
        (pl.col(LABEL_COL) == 0) &  # BGL: label = 0 (normal)
        (pl.col("EventTemplate").is_not_null())
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
    
    Strategy (MANTIDO de 02_loggpt_small):
    1. Group logs by 'node_id' (Session) - ADAPTADO: node_id ao invés de test_id
    2. Concatenate 'EventTemplate' (text) with newline separator.
    3. Tokenize the full session text.
    4. Chunk into 'block_size' for training.
    """
    path = data_path or DATA_PATH
    df = load_bgl_data(path)
    
    # Group by node_id and concat templates
    # Polars is fast
    print(f"🔄 Grouping logs by session ({SESSION_ID_COL})...")
    
    # We want a list of strings, where each string is a full session
    # "Template1 \n Template2 \n Template3"
    
    # Fallback to pandas for complex groupby string  agg if polars is tricky with list join on older versions
    # But Polars 'implode' or 'agg' works.
    
    # Using 'EventTemplate' (clean text)
    # CRITICAL FIX (P1.4): Ensure global sort by timestamp before grouping!
    sessions = (
        df.sort("timestamp")  # MANTIDO
        .group_by(SESSION_ID_COL)  # ADAPTADO: node_id
        .agg(pl.col("EventTemplate"))
        .select("EventTemplate")
    )
    
    # Convert to list of strings
    # Join with [SEP] or \n
    text_sessions = []
    
    rows = sessions.rows() # List of tuples
    for row in rows:
        # row[0] is the list of strings for that session
        templates = row[0]
        
        # NOVO: Limitar sessões muito longas (BGL node_id pode ter milhares de logs)
        if len(templates) > SESSION_MAX_LOGS:
            # Dividir em sub-sessões de SESSION_MAX_LOGS
            for i in range(0, len(templates), SESSION_MAX_LOGS):
                sub_session = templates[i:i+SESSION_MAX_LOGS]
                session_text = " \n ".join(sub_session)
                text_sessions.append(session_text)
        else:
            session_text = " \n ".join(templates)
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
        num_proc=1, 
        remove_columns=["text"]
    )
    
    # Grouping Function (Chunking)
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder
        total_length = (total_length // block_size) * block_size
        
        # Split by chunks of block_size
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
        batch_size=1000,
        num_proc=4,
    )
    
    return lm_datasets

if __name__ == "__main__":
    # Test run
    from transformers import AutoTokenizer
    # Use distinct-distilgpt2 or similar small model
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # distilgpt2 doesn't have pad token usually
    tokenizer.pad_token = tokenizer.eos_token
    
    ds = prepare_llm_dataset(tokenizer)
    print(ds)
