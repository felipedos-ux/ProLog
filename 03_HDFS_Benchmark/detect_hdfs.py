import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, AutoTokenizer
import polars as pl
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import config_hdfs as config
from model import LogGPT
from utils.logger import setup_logger

logger = setup_logger(__name__)

class HDFSTestDataset(Dataset):
    def __init__(self, df, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = df
        # df columns: BlockId, label, EventTemplate (str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.row(idx, named=True)
        seq_str = row['EventTemplate']
        label = row['label']
        blk = row['BlockId']
        
        # Tokenize
        # Note: We tokenize the WHOLE sequence.
        # LogGPT needs context.
        tokens = self.tokenizer.encode(seq_str)
        
        # Truncate if too long
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
            
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'block_id': blk,
            'seq_len': len(tokens)
        }

def collate_fn(batch):
    # Dynamic padding
    max_len = max(x['seq_len'] for x in batch)
    
    padded_ids = torch.full((len(batch), max_len), 50256, dtype=torch.long)
    labels = []
    block_ids = []
    
    for i, x in enumerate(batch):
        l = x['seq_len']
        padded_ids[i, :l] = x['input_ids']
        labels.append(x['label'])
        block_ids.append(x['block_id'])
        
    return {
        'input_ids': padded_ids,
        'label': torch.tensor(labels, dtype=torch.long),
        'block_id': block_ids
    }

def detect():
    result_path = config.DATA_DIR / "HDFS_test_results.csv"
    test_path = config.DATA_DIR / "HDFS_test.csv"
    model_path = config.MODEL_DIR / "hdfs_loggpt.pt"
    
    if not test_path.exists():
        logger.error(f"{test_path} not found.")
        return

    logger.info("Loading Data & Model...")
    df = pl.read_csv(str(test_path))
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Config (Saved or Manual)
    # We use config_hdfs.py defaults
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        n_embd=config.N_EMBD,
        dropout=config.DROPOUT
    )
    model_config.block_size = config.BLOCK_SIZE
    
    model = LogGPT(model_config)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    
    dataset = HDFSTestDataset(df, tokenizer, config.BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Detection
    # Logic: For each step t, predict t+1. Check if true t+1 is in Top-K.
    # If ANY step is anomalous => Block is Anomalous.
    
    K = 5
    results = [] # (BlockId, TrueLabel, PredLabel, FirstAnomalyStep)
    
    logger.info(f"Running Detection (Top-{K})...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(config.DEVICE)
            # labels (anom/norm)
            batch_labels = batch['label'].cpu().numpy()
            block_ids = batch['block_id']
            
            # Forward
            # Input: [B, T]
            # Output Logits: [B, T, V]
            # We predict for positions 0..T-2 to match 1..T-1?
            # Actually: Input x_0..x_{T-1}.
            # Logits at pos t predict x_{t+1}.
            # We check prediction at t against x_{t+1}.
            
            logits, _ = model(input_ids)
            # logits: [B, T, V]
            
            # We only care about predictions where target exists.
            # Targets are input_ids[:, 1:]
            targets = input_ids[:, 1:]
            # Predictions are logits[:, :-1, :]
            preds = logits[:, :-1, :]
            
            # Check Top-K
            # preds: [B, T-1, V]
            # targets: [B, T-1]
            
            # Use torch.topk
            probs = torch.softmax(preds, dim=-1)
            topk_vals, topk_inds = torch.topk(probs, K, dim=-1)
            # topk_inds: [B, T-1, K]
            
            # Check if target is in topk
            # targets.unsqueeze(-1): [B, T-1, 1]
            # match: [B, T-1, K] boolean
            matches = (topk_inds == targets.unsqueeze(-1)).any(dim=-1)
            # matches[b, t] = True if correct prediction, False if Anomaly
            
            # Mask out Padding?
            # targets == 50256 (pad_id) should be ignored.
            target_mask = (targets != 50256)
            
            # valid_anomalies = (~matches) & target_mask
            valid_anomalies = (~matches) & target_mask
            
            # Check if ANY anomaly in sequence
            is_anom_pred = valid_anomalies.any(dim=1).cpu().numpy() # [B]
            
            # Find first anomaly index (Lead Time proxy)
            # Argmax returns first True
            first_indices = valid_anomalies.int().argmax(dim=1).cpu().numpy()
            # If no anomaly, argmax returns 0 (ambiguous if 0 is anomaly)
            # but is_anom_pred handles existence.
            
            for i in range(len(block_ids)):
                blk = block_ids[i]
                true_lbl = batch_labels[i]
                pred_lbl = 1 if is_anom_pred[i] else 0
                first_step = first_indices[i] if is_anom_pred[i] else -1
                
                results.append((blk, true_lbl, pred_lbl, first_step))

    # Metrics
    y_true = [r[1] for r in results]
    y_pred = [r[2] for r in results]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true, y_pred)
    
    logger.info(f"Results (Top-{K}):")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Accuracy: {acc:.4f}")
    
    # Save Results
    res_df = pl.DataFrame(results, schema=["BlockId", "Label", "Predicted", "FirstAnomalyStep"], orient="row")
    res_df.write_csv(str(result_path))
    logger.info(f"Detailed results saved to {result_path}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    detect()
