# -*- coding: utf-8 -*-
"""
SIAT Dataset Loader
===================
Custom dataset and tokenizer for SIAT logs.
Treats each normalized endpoint as a single token.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from config import SESSION_DATA, BLOCK_SIZE

class SimpleTokenizer:
    """Maps unique endpoints to integers."""
    def __init__(self):
        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        self.itos = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def fit(self, sessions_list):
        """Builds vocab from list of sessions (lists of strings)."""
        unique_tokens = set()
        for session in sessions_list:
            unique_tokens.update(session)
            
        # Sort for determinism
        for token in sorted(unique_tokens):
            if token not in self.stoi:
                self.stoi[token] = self.vocab_size
                self.itos[self.vocab_size] = token
                self.vocab_size += 1
        print(f"Tokenizer fitted. Vocab size: {self.vocab_size}")

    def encode(self, session):
        """Converts list of strings to list of ints."""
        return [self.stoi.get(token, self.stoi['<UNK>']) for token in session]

    def decode(self, ids):
        """Converts list of ints back to strings."""
        return [self.itos.get(i, '<UNK>') for i in ids]

class SIATDataset(Dataset):
    def __init__(self, sessions_df, tokenizer, block_size=128, mode='train'):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode
        
        # Filter: Train only on NORMAL sessions (label=0)
        # Test: Keep all
        if mode == 'train':
            self.data = sessions_df[sessions_df['label'] == 0]['events'].tolist()
            print(f"[{mode}] Filtered normal sessions: {len(self.data)}")
        else:
            self.data = sessions_df['events'].tolist()
            self.labels = sessions_df['label'].tolist()
            print(f"[{mode}] Loaded all sessions: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        session = self.data[idx]
        
        # Encode
        input_ids = self.tokenizer.encode(session)
        
        # Truncate/Pad
        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
        else:
            padding = [0] * (self.block_size - len(input_ids))
            input_ids = input_ids + padding
            
        x = torch.tensor(input_ids, dtype=torch.long)
        
        if self.mode == 'train':
            # For causal LM, labels are same as input (shifted inside model)
            # Mask padding in loss? (Optionally, but simple cross entropy ignores index -100)
            # Here we just return x as both input and target source
            return {"input_ids": x, "labels": x}  # HuggingFace style collator handles shift? 
                                                 # No, we do custom loop. Return x.
        else:
            # For eval, we need the anomaly label
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return {"input_ids": x, "anom_label": y}

def load_data():
    print(f"📦 Loading {SESSION_DATA}...")
    with open(SESSION_DATA, "rb") as f:
        data = pickle.load(f)
    
    # 1. Fit tokenizer on TRAIN data (Normal + Anomalous to know vocab? 
    # Actually, usually fit on everything to not crash on unknown tests, 
    # but strictly should fit on Train. UNK handles the rest.)
    # Let's fit on ALL train data (including anomalous sessions present in train split if any? 
    # Wait, preprocess split train/test stratified.
    # Train split has normal AND anomalous.
    # But we only train model on NORMAL.
    # However, tokenizer should know about all tokens in TRAIN partition at least.
    
    tokenizer = SimpleTokenizer()
    tokenizer.fit(data['train']['events'].tolist())
    
    train_dataset = SIATDataset(data['train'], tokenizer, BLOCK_SIZE, mode='train')
    
    # For test, use same tokenizer (might have UNKs if test has new endpoints)
    test_dataset = SIATDataset(data['test'], tokenizer, BLOCK_SIZE, mode='test')
    
    return train_dataset, test_dataset, tokenizer
