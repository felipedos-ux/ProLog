import polars as pl

def check_start_logs():
    print("Loading data...")
    df = pl.read_csv("../data/OpenStack_data_original.csv", infer_schema_length=10000)
    
    # Group by session
    # We want to see the first log of each session
    print("Grouping...")
    first_logs = (
        df.sort("timestamp")
        .group_by("test_id")
        .agg([
            pl.col("EventTemplate").first().alias("first_log"),
            pl.col("anom_label").max().alias("label")
        ])
    )
    
    # Check frequency of "Workload started!"
    target_log = "Workload started! Create image <*>"
    
    # Filter for target
    # Note: EventTemplate might vary slightly, so we use string contains or just check top 5
    
    print("\nTop 5 First Logs for NORMAL sessions:")
    normal_starts = (first_logs.filter(pl.col("label") == 0)
                     .group_by("first_log")
                     .len()
                     .sort("len", descending=True)
                     .head(5))
    print(normal_starts)
    
    print("\nTop 5 First Logs for ANOMALY sessions:")
    anom_starts = (first_logs.filter(pl.col("label") == 1)
                     .group_by("first_log")
                     .len()
                     .sort("len", descending=True)
                     .head(5))
    print(anom_starts)
    
    # Check specific target overlap
    print(f"\nChecking overlap for '{target_log}':")
    norm_count = first_logs.filter((pl.col("label") == 0) & (pl.col("first_log").str.contains("Workload started"))).height
    anom_count = first_logs.filter((pl.col("label") == 1) & (pl.col("first_log").str.contains("Workload started"))).height
    
    print(f"Normal sessions starting with 'Workload started': {norm_count}")
    print(f"Anomaly sessions starting with 'Workload started': {anom_count}")

if __name__ == "__main__":
    check_start_logs()
