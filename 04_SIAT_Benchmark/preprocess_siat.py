# -*- coding: utf-8 -*-
"""
SIAT Preprocessing Pipeline
===========================
1. Loads raw CSV (latin-1)
2. Groups by IP + 5min window -> Session ID
3. Drops sensitive columns (IP, City, Country, Browser)
4. Normalizes endpoints (regex)
5. Splits into Train/Test
6. Saves as siat_sessions.pkl
"""
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

# Config
INPUT_FILE = "D:/ProLog/data/siat.csv"
OUTPUT_DIR = "D:/ProLog/04_SIAT_Benchmark/data"
SESSION_WINDOW = 300  # 5 minutes in seconds
TEST_SIZE = 0.2
SEED = 42

def normalize_endpoint(endpoint):
    """Normalizes endpoints to reduce vocabulary size."""
    if not isinstance(endpoint, str):
        return "UNKNOWN"
    
    # Lowercase
    endpoint = endpoint.lower()
    
    # 1. UUIDs (e.g., a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11)
    endpoint = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '{uuid}', endpoint)
    
    # 2. Integers (IDs)
    endpoint = re.sub(r'/\d+', '/{id}', endpoint)
    
    # 3. Query params (remove everything after ?)
    endpoint = endpoint.split('?')[0]
    
    # 4. Specific known patterns (optional refinement)
    # e.g. dates YYYY-MM-DD
    endpoint = re.sub(r'\d{4}-\d{2}-\d{2}', '{date}', endpoint)
    
    return endpoint

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"[1/6] Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='latin-1', header=None, low_memory=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Name columns based on analysis
    df.columns = ['timestamp', 'status_code', 'method', 'endpoint', 'service', 'server', 'ip', 'city', 'country', 'user_agent']
    
    # Fix timestamp
    print("[2/6] Processing timestamps...")
    if df['timestamp'].iloc[0] == 'timestamp':
        df = df.iloc[1:].reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Group sessions (IP + Window)
    print("[3/6] Identifying sessions...")
    df = df.sort_values(['ip', 'timestamp']).reset_index(drop=True)
    df['time_diff'] = df.groupby('ip')['timestamp'].diff().dt.total_seconds()
    df['new_session'] = (df['time_diff'].isna()) | (df['time_diff'] > SESSION_WINDOW)
    df['session_id'] = df.groupby('ip')['new_session'].cumsum()
    df['global_session_id'] = df['ip'].astype(str) + '_' + df['session_id'].astype(str)
    
    # Remove sensitive info
    print("[4/6] Anonymizing data...")
    # Kept: timestamp, status_code, method, endpoint, global_session_id
    # Dropped: ip, city, country, user_agent, service, server (unless needed)
    df_clean = df[['timestamp', 'global_session_id', 'method', 'endpoint', 'status_code']].copy()
    
    # Normalize endpoints
    print("[5/6] Normalizing endpoints...")
    df_clean['endpoint_norm'] = df_clean['endpoint'].apply(normalize_endpoint)
    
    # Combine Method + Endpoint for "Event Token"
    # e.g., "GET /api/user/{id}"
    df_clean['event_token'] = df_clean['method'] + " " + df_clean['endpoint_norm']
    
    # Define session label (Anomalous if any 4xx/5xx)
    # Using status_code as int
    df_clean['status_code'] = pd.to_numeric(df_clean['status_code'], errors='coerce').fillna(200).astype(int)
    df_clean['is_error'] = df_clean['status_code'] >= 400
    
    # Aggregation
    print("      Aggregating sessions...")
    sessions = df_clean.groupby('global_session_id').agg({
        'event_token': list,
        'timestamp': lambda x: [t.isoformat() for t in x],
        'is_error': 'max'  # If any error -> session is anomalous
    }).reset_index()
    
    sessions.rename(columns={'event_token': 'events', 'is_error': 'label'}, inplace=True)
    
    # Stats
    n_total = len(sessions)
    n_anom = sessions['label'].sum()
    print(f"      Total Sessions: {n_total}")
    print(f"      Anomalous: {n_anom} ({n_anom/n_total*100:.2f}%)")
    print(f"      Vocab Size: {df_clean['event_token'].nunique()}")
    
    # Split
    print("[6/6] Splitting Train/Test...")
    train, test = train_test_split(sessions, test_size=TEST_SIZE, random_state=SEED, stratify=sessions['label'])
    
    # Save
    save_path = Path(OUTPUT_DIR) / "siat_sessions.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({'train': train, 'test': test}, f)
        
    print(f"✅ Saved to {save_path}")

if __name__ == "__main__":
    main()
