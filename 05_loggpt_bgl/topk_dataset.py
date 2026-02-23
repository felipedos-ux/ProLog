"""
Dataset para treinar LogGPT com IDs de templates (não BPE)

Cada template é tratado como um token único, reduzindo o vocabulário
de 50k (GPT-2) para ~320 (templates BGL).
"""

import torch
from torch.utils.data import Dataset
import polars as pl
import json
from pathlib import Path
from typing import List

class BGLTemplateDataset(Dataset):
    """Dataset de sequências BGL usando IDs de templates"""
    
    def __init__(self, parquet_path: str, vocab_path: str, block_size: int = 20, mode="train"):
        """
        Args:
            parquet_path: Caminho para arquivo .parquet (train/val/test)
            vocab_path: Caminho para bgl_template_vocab.json
            block_size: Tamanho máximo da sequência (em templates)
            mode: "train", "val" ou "test"
        """
        self.block_size = block_size
        self.mode = mode
        
        # Carregar vocabulário
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.template_to_id = vocab_data['template_to_id']
        self.vocab_size = vocab_data['vocab_size']
        self.pad_id = self.template_to_id["<PAD>"]
        self.unk_id = self.template_to_id["<UNK>"]
        
        # Carregar dados
        df = pl.read_parquet(parquet_path)
        
        # Filtrar apenas normais para treino
        if mode == "train":
            df = df.filter(pl.col("label") == 0)
        
        self.sequences = []
        self.labels = []
        
        for row in df.iter_rows(named=True):
            templates = row['sequence']  # Lista de strings
            label = row['label']
            
            # Converter templates para IDs
            ids = []
            for template in templates:
                if template in self.template_to_id:
                    ids.append(self.template_to_id[template])
                else:
                    ids.append(self.unk_id)
            
            self.sequences.append(ids)
            self.labels.append(label)
        
        print(f"{mode.upper()} dataset: {len(self.sequences)} sequences, vocab_size={self.vocab_size}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Ajustar tamanho para block_size
        if len(sequence) > self.block_size:
            # Truncar
            sequence = sequence[:self.block_size]
        else:
            # Pad
            sequence = sequence + [self.pad_id] * (self.block_size - len(sequence))
        
        # Converter para tensor
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)  # [0, 1, 2, ..., n-1]
        targets = torch.tensor(sequence[1:], dtype=torch.long)     # [1, 2, 3, ..., n]
        
        return {
            'input_ids': input_ids,
            'targets': targets,
            'label': label
        }
