import polars as pl
import json
from pathlib import Path
from tqdm import tqdm
import config_count as config

# PATHS
DATA_FILE = config.DATA_FILE
OUTPUT_FILE = config.MODEL_DIR / "template_freq.json"

def generate_frequency_map():
    print(f"🚀 Loading Training Data from {DATA_FILE}...")
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Run dataset_bgl_count.py first.")
        
    df = pl.read_csv(str(DATA_FILE), infer_schema_length=10000)
    
    print(f"📊 Analyzing {len(df)} logs...")
    
    # Count Templates
    counts = df["EventTemplate"].value_counts()
    
    # Convert to Dict
    freq_map = {}
    total_logs = len(df)
    
    print("⚡ Calculating Frequencies...")
    for row in tqdm(counts.iter_rows(named=True), total=len(counts)):
        template = str(row["EventTemplate"])
        count = row["count"]
        freq = count / total_logs
        
        freq_map[template] = {
            "count": count,
            "freq": freq
        }
        
    # Stats
    print(f"🔹 Unique Templates: {len(freq_map)}")
    
    # Save
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(freq_map, f, indent=4)
        
    print("✅ Frequency Map Generated.")

if __name__ == "__main__":
    generate_frequency_map()
