"""
Custom Event Tokenizer for LogGPT.

Maps each unique EventId to a single integer token.
This is the standard approach in log anomaly detection papers:
- Each log event type = 1 token (not subwords)
- Model learns EVENT SEQUENCES, not text patterns
- Unseen events get a special UNK token (high surprise = anomaly signal)
"""
import polars as pl
import json
import os
from typing import List, Dict, Optional
from pathlib import Path

from config import DATA_FILE, LOG_COLUMN


class EventTokenizer:
    """Simple tokenizer that maps EventIds to integer token IDs."""
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SEP_TOKEN = "<SEP>"  # Separator between logs in a session
    
    def __init__(self):
        self.event_to_id: Dict[str, int] = {}
        self.id_to_event: Dict[int, str] = {}
        self.vocab_size: int = 0
        
        # Reserve special tokens
        self._add_token(self.PAD_TOKEN)  # 0
        self._add_token(self.UNK_TOKEN)  # 1
        self._add_token(self.SEP_TOKEN)  # 2
        
    def _add_token(self, token: str) -> int:
        if token not in self.event_to_id:
            idx = len(self.event_to_id)
            self.event_to_id[token] = idx
            self.id_to_event[idx] = token
            self.vocab_size = len(self.event_to_id)
        return self.event_to_id[token]
    
    def fit(self, events: List[str]) -> "EventTokenizer":
        """Build vocabulary from a list of event IDs (training data only)."""
        unique_events = sorted(set(str(e) for e in events if e is not None))
        for event in unique_events:
            self._add_token(event)
        print(f"📊 EventTokenizer: {self.vocab_size} tokens "
              f"({len(unique_events)} events + 3 special)")
        return self
    
    @property
    def pad_token_id(self) -> int:
        return self.event_to_id[self.PAD_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.event_to_id[self.UNK_TOKEN]
    
    @property
    def sep_token_id(self) -> int:
        return self.event_to_id[self.SEP_TOKEN]
    
    def encode(self, events: List[str], add_sep: bool = True) -> List[int]:
        """
        Encode a list of event IDs into token IDs.
        
        Args:
            events: List of EventId strings
            add_sep: Whether to add SEP token between events
            
        Returns:
            List of integer token IDs
        """
        ids = []
        for i, event in enumerate(events):
            event_str = str(event) if event is not None else self.UNK_TOKEN
            token_id = self.event_to_id.get(event_str, self.unk_token_id)
            ids.append(token_id)
            if add_sep and i < len(events) - 1:
                ids.append(self.sep_token_id)
        return ids
    
    def encode_single(self, event: str) -> int:
        """Encode a single event ID."""
        return self.event_to_id.get(str(event), self.unk_token_id)
    
    def decode(self, ids: List[int]) -> List[str]:
        """Decode token IDs back to event strings."""
        return [self.id_to_event.get(i, self.UNK_TOKEN) for i in ids]
    
    def save(self, path: str):
        """Save tokenizer vocab to JSON."""
        with open(path, "w") as f:
            json.dump({
                "event_to_id": self.event_to_id,
                "vocab_size": self.vocab_size,
            }, f, indent=2)
        print(f"💾 EventTokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "EventTokenizer":
        """Load tokenizer vocab from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls()
        tok.event_to_id = data["event_to_id"]
        tok.id_to_event = {int(v): k for k, v in tok.event_to_id.items()}
        tok.vocab_size = data["vocab_size"]
        print(f"📦 EventTokenizer loaded: {tok.vocab_size} tokens")
        return tok


def build_tokenizer_from_training_data(data_path: Path = DATA_FILE) -> EventTokenizer:
    """Build and save an EventTokenizer from the NORMAL training data only."""
    df = pl.read_csv(str(data_path), infer_schema_length=10000)
    
    # Only normal events for vocabulary
    normal_events = df.filter(
        (pl.col("anom_label") == 0) & 
        (pl.col(LOG_COLUMN).is_not_null())
    )[LOG_COLUMN].to_list()
    
    tokenizer = EventTokenizer()
    tokenizer.fit(normal_events)
    
    # Save alongside model
    from config import MODEL_DIR
    save_path = os.path.join(str(MODEL_DIR), "event_tokenizer.json")
    tokenizer.save(save_path)
    
    return tokenizer


if __name__ == "__main__":
    tok = build_tokenizer_from_training_data()
    
    # Test
    print(f"\nVocab size: {tok.vocab_size}")
    print(f"PAD={tok.pad_token_id}, UNK={tok.unk_token_id}, SEP={tok.sep_token_id}")
    
    # Test encoding
    sample_events = ["E1", "E2", "E3", "NEVER_SEEN_EVENT"]
    encoded = tok.encode(sample_events)
    decoded = tok.decode(encoded)
    print(f"\nSample: {sample_events}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"UNK check: 'NEVER_SEEN_EVENT' -> {tok.encode_single('NEVER_SEEN_EVENT')} (should be {tok.unk_token_id})")
