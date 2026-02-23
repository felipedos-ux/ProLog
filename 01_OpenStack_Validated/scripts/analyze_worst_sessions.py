"""
Script to analyze sessions with negative lead times.
Goal: Understand why detection happened AFTER the failure.
"""
import polars as pl
from config import DATA_FILE

df = pl.read_csv(str(DATA_FILE), infer_schema_length=10000)

# Sessions with worst lead times (from the image)
worst_sessions = [6, 12, 3, 19, 32, 31, 18, 24, 21, 29]

print("="*70)
print("ANALYSIS: SESSIONS WITH NEGATIVE LEAD TIME")
print("="*70)

for tid in worst_sessions[:5]:  # Analyze first 5 in detail
    session = df.filter(pl.col('test_id') == tid)
    ts_col = "time_hour" if "time_hour" in session.columns else "timestamp"
    session = session.sort(ts_col)
    
    print(f"\n{'='*60}")
    print(f"SESSION {tid}")
    print(f"{'='*60}")
    print(f"Total Logs: {len(session)}")
    print(f"Anomaly Label: {session['anom_label'][0]}")
    
    templates = session['EventTemplate'].to_list()
    timestamps = session[ts_col].to_list()
    
    print(f"\n📋 First 5 logs:")
    for i in range(min(5, len(templates))):
        print(f"  [{i}] {templates[i][:70]}...")
    
    print(f"\n📋 Last 5 logs:")
    for i in range(max(0, len(templates)-5), len(templates)):
        print(f"  [{i}] {templates[i][:70]}...")
    
    # Calculate time span
    import pandas as pd
    try:
        ts_start = pd.to_datetime(timestamps[0])
        ts_end = pd.to_datetime(timestamps[-1])
        duration = (ts_end - ts_start).total_seconds() / 60
        print(f"\n⏱️ Session Duration: {duration:.2f} minutes")
    except:
        print(f"\n⏱️ Session Duration: Could not parse timestamps")
    
    # Unique templates in session
    unique_templates = set(templates)
    print(f"🔢 Unique Log Templates: {len(unique_templates)}")

print("\n" + "="*70)
print("HYPOTHESIS FOR NEGATIVE LEAD TIMES:")
print("="*70)
print("""
1. SUDDEN CRASHES: Some failures (auth key error) happen instantaneously
   without precursor warning logs. The first anomalous log IS the failure.

2. SHORT SESSIONS: Sessions with very few logs don't have enough context
   for the model to detect patterns before the failure.

3. SIMILAR PATTERNS: "Attach volume" failures have similar log patterns
   to normal operations until the very last moment.

POTENTIAL SOLUTIONS:
- Multi-modal: Combine with system metrics (CPU, memory, network)
- Ensemble: Use rule-based detection for known crash patterns
- Lookahead: Train model to predict N+2, N+3 tokens ahead
- Session features: Analyze session-level statistics (duration, frequency)
""")
