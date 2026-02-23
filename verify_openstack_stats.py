import polars as pl
from pathlib import Path

DATA_PATH = "D:/ProLog/data/OpenStack_data_original.csv"

def main():
    print(f"📦 Loading OpenStack data from {DATA_PATH}...")
    try:
        df = pl.read_csv(DATA_PATH, infer_schema_length=0)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Total rows: {len(df)}")
    
    # Check columns
    print(f"Columns: {df.columns}")
    
    # Ensure correct types
    if "anom_label" not in df.columns:
        print("Error: 'anom_label' column not found")
        # Try to guess or print head
        print(df.head())
        return

    # Convert anom_label to int
    df = df.with_columns(pl.col("anom_label").cast(pl.Int32, strict=False))

    # Group by test_id (Session)
    if "test_id" in df.columns:
        print("Grouping by test_id...")
        
        # Determine if a session is anomalous: max(anom_label) > 0
        sessions = df.group_by("test_id").agg([
            pl.col("anom_label").max().alias("is_anomalous"),
            pl.count().alias("log_count")
        ])
        
        total_sessions = len(sessions)
        anomalous_sessions = sessions.filter(pl.col("is_anomalous") > 0).height
        normal_sessions = total_sessions - anomalous_sessions
        
        print(f"\n=== OPENSTACK STATS ===")
        print(f"Total Log Lines: {len(df):,}")
        print(f"Total Sessions: {total_sessions:,}")
        print(f"Normal Sessions: {normal_sessions:,}")
        print(f"Anomalous Sessions: {anomalous_sessions:,}")
        print(f"Anomaly Rate: {anomalous_sessions / total_sessions * 100:.2f}%")
        print(f"Avg Session Size: {sessions['log_count'].mean():.2f}")
    else:
        print("Error: 'test_id' column not found.")

if __name__ == "__main__":
    main()
