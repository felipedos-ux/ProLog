import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from transformers import AutoTokenizer

class HDFSContrastiveDataset(Dataset):
    """
    Dataset for Self-Supervised Contrastive Learning on HDFS using Subwords (DistilGPT-2).
    Generates two augmented views (positive pairs) for each normal session.
    """
    def __init__(self, data_path, model_name="distilgpt2", block_size=1024, max_sessions=5000, 
                 mask_prob=0.15, swap_prob=0.05):
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.swap_prob = swap_prob
        
        # Tokenizer setup
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.eos_token_id
        
        # GPT2 has no strict [MASK] so we fallback to EOS or random replacement 
        # (Using EOS behaves similarly to dropping/masking the contextual signal of that subword)
        self.mask_token_id = self.tokenizer.eos_token_id
        
        # Load data
        print(f"Loading train data from {data_path}...")
        df = pd.read_csv(data_path)
        
        if 'anom_label' in df.columns:
            df = df[df['anom_label'] == 0]
            
        print(f"Loaded {len(df)} normal log rows for training.")
        
        print("Grouping logs by session (session_id)...")
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            
        # Concat templates as done in original prepare_llm_dataset
        session_groups = df.groupby('session_id')['EventTemplate'].apply(lambda x: " \n ".join(x.astype(str))).to_dict()
        
        if max_sessions is not None:
            session_ids = list(session_groups.keys())[:max_sessions]
            session_groups = {k: session_groups[k] for k in session_ids}
            
        print(f"Sessions to train: {len(session_groups)}")
        
        # Tokenize sessions into long documents
        documents = []
        for session_id, text in session_groups.items():
            tokens = self.tokenizer(text)['input_ids']
            if len(tokens) > 0:
                documents.append(tokens)
                
        # Chunk into block_size
        self.chunks = []
        for doc in documents:
            for i in range(0, len(doc), self.block_size):
                chunk = doc[i:i + self.block_size]
                if len(chunk) > 10: # Only keep meaningful chunks
                    self.chunks.append(chunk)
                    
        print(f"Total chunks created: {len(self.chunks)}")

    def augment(self, chunk):
        """
        Applies random masking and token swapping to create an augmented view.
        """
        aug_chunk = chunk.copy()
        
        # 1. Random Masking
        for i in range(len(aug_chunk)):
            if random.random() < self.mask_prob:
                aug_chunk[i] = self.mask_token_id
                
        # 2. Token Swapping (Local)
        for i in range(len(aug_chunk) - 1):
            if random.random() < self.swap_prob:
                # Swap with adjacent token
                aug_chunk[i], aug_chunk[i+1] = aug_chunk[i+1], aug_chunk[i]
                
        return aug_chunk

    def pad_chunk(self, chunk):
        """ Pads or truncates to block_size """
        if len(chunk) > self.block_size:
            chunk = chunk[:self.block_size]
        else:
            chunk = chunk + [self.pad_token_id] * (self.block_size - len(chunk))
        return torch.tensor(chunk, dtype=torch.long)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        base_chunk = self.chunks[idx]
        
        # View 1: Mild augmentation or original
        view1 = self.augment(base_chunk) if random.random() < 0.8 else base_chunk
        
        # View 2: Aggressive augmentation
        view2 = self.augment(base_chunk) 
        
        # For standard language modeling (predict next token) we still need inputs and targets
        # We will use view1 for LM task
        x = self.pad_chunk(view1[:-1])
        y = self.pad_chunk(view1[1:])
        
        # For contrastive learning, we return both padded block_size views
        v1_padded = self.pad_chunk(view1)
        v2_padded = self.pad_chunk(view2)
        
        return {
            'input_ids': x,
            'labels': y,
            'view1': v1_padded,
            'view2': v2_padded
        }

def get_dataloaders(data_path, batch_size=16, block_size=1024, max_sessions=5000, val_split=0.1):
    dataset = HDFSContrastiveDataset(data_path, block_size=block_size, max_sessions=max_sessions)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, dataset.tokenizer.vocab_size
