import polars as pl
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import config_hdfs as config
from utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_regex(template):
    """Converts Loghub template to Regex."""
    # Template format: "Receiving block <*> src: <*> dest: <*>"
    # <*> is wildcard.
    if pd.isna(template):
        return r".*"
    
    # Escape special regex chars in the text parts
    parts = template.split('<*>')
    escaped_parts = [re.escape(p.strip()) for p in parts]
    
    # Helper to handle spaces around wildcards if needed, 
    # but simple .*? join usually works for Loghub templates
    # We use .*? for non-greedy match of variable parts
    # Add ^ and $ to ensure full match? Or partial?
    # Loghub parsers usually match full content.
    
    # Refined: join with .*?
    # Note: Empty parts mean the wildcard was at start/end or repeated.
    # If part is empty string, re.escape("") is "".
    
    regex = r".*?".join(escaped_parts)
    
    # Handle start/end hooks
    # If template started with <*>, the split has empty string at index 0.
    # r"" + .*? + "text" -> .*?text
    
    # We accept partial match inside the log content?
    # The "Content" extracted from log line usually matches the template exactly (modulo variables)
    # But let's allow loose matching to be safe.
    
    return regex

def process_dataset():
    logger.info("Starting HDFS Raw Log Processing (Polars Optimized)...")
    
    # 1. Load Templates
    if not config.TEMPLATE_FILE.exists():
        logger.error(f"Template file not found: {config.TEMPLATE_FILE}")
        return
        
    templates_df = pl.read_csv(str(config.TEMPLATE_FILE))
    # Columns: EventId, EventTemplate
    logger.info(f"Loaded {len(templates_df)} templates.")
    
    # 2. Build Regex-EventId Map
    # Sort templates by length? Specificity?
    # In 'when-then' chain, order matters. First match wins.
    # Usually we want specific ones first.
    # But for HDFS, templates are quite distinct. All good.
    
    template_list = templates_df.to_dicts()
    
    # 3. Load Raw Logs
    log_path = config.LOG_FILE
    if not log_path.exists():
        logger.error(f"Log file not found: {log_path}")
        return

    logger.info(f"Scanning Raw Logs from {log_path}...")
    
    # Read as single column "raw"
    # Polars scan_csv with separator that doesn't exist to get full lines?
    # Or read_text? HDFS 1.5GB is fit in RAM (machine has 32GB?).
    # scan_csv(separator='\0') works effectively as read lines if no null char.
    # Or use read_csv with `has_header=False`, `new_columns=['raw_line']`, `separator='\u0001'` (unlikely char)
    
    # Better: Use `pl.scan_csv` with a non-existent separator
    q = pl.scan_csv(
        str(log_path),
        has_header=False,
        new_columns=["raw_line"],
        separator="\x00", # Non-existent byte (Null)
        quote_char=None,  # Disable quoting to avoid multiline issues
        ignore_errors=True,
        truncate_ragged_lines=False
    )
    
    # 4. Extract BlockId and Content
    # HDFS Log Format:
    # 081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 ...
    # We want "Receiving block blk_... ..." as Content? 
    # Or is "dfs.DataNode$DataXceiver" part of it?
    # Templates usually match the message AFTER the component/level.
    # Template: "Receiving block <*> src: <*> dest: <*>"
    # So we need to extract everything after the standard header.
    # Header: "Date Time Pid Level Component: "
    # Regex: `^.{6} .{6} \d+ \w+ .*?: (.*)$`
    # Let's try simpler: Split by ": " and take the last part? 
    # But content might contain ": ".
    # The Loghub parser usually uses fixed indices or regex.
    # HDFS logs: The content starts after the first ": " ?? 
    # "INFO dfs.DataNode$DataXceiver: Receiving block ..." -> Split on ": " -> [Head, Content]
    
    # AND BlockId: `blk_[-0-9]+`
    
    logger.info("Parsing lines (BlockId & Content extraction)...")
    
    q = q.with_columns([
        pl.col("raw_line").str.extract(r"(blk_-?\d+)", 1).alias("BlockId"),
        # Content extraction: Text after the first occurrence of ": " (plus 1 char for space?)
        # Or just extract extracting regex: `\d+ \w+ .*?: (.*)`
        pl.col("raw_line").str.replace(r"^.+?:\s+", "").alias("Content") 
    ])
    
    # Filter valid blocks
    q = q.filter(pl.col("BlockId").is_not_null())
    
    # 5. Apply Template Matching (The Big Switch)
    # We construct a massive 'when().then()...otherwise(Unknown)' expression
    
    logger.info("Building Template Matching Expression...")
    
    # optimization: compile regexes in python, but here we construct polars expr
    # We iterate templates.
    match_expr = pl.lit("Unknown")
    
    # We iterate in reverse so the LAST added 'when' is at the top? 
    # No, `when(c1).then(r1).otherwise(when(c2).then(r2)...)` matches c1 first.
    # So we iterate normally.
    
    # Constructing the chain:
    # start with `otherwise("Unknown")`
    # wrap with `when(match).then(id).otherwise(...)`
    
    # To avoid deep recursion limit in Python, we can't just chain .otherwise() 30 times?
    # Chains of 30 are fine.
    
    current_expr = pl.lit("E_Unknown")
    
    # Process templates (maybe sort by complexity?)
    for t in reversed(template_list): # Reverse so the first in list ends up at outer layer? 
        # Wait. 
        # Expr = when(cond).then(val).otherwise(Expr)
        # If I want T1 to be checked first:
        # Expr = when(T1).then(E1).otherwise( when(T2).then(E2)... )
        # So I should start from End (Unknown) and wrap backwards to T1.
        
        eid = t['EventId']
        tmpl = t['EventTemplate']
        rgx = generate_regex(tmpl)
        
        current_expr = pl.when(
            pl.col("Content").str.contains(rgx)
        ).then(pl.lit(eid)).otherwise(current_expr)
        
    q = q.with_columns(current_expr.alias("EventId"))
    
    # 6. Group By Session
    logger.info("Grouping sessions...")
    # Select only needed columns
    sessions_lazy = (
        q.select(["BlockId", "EventId"])
        .group_by("BlockId")
        .agg(pl.col("EventId"))
    )
    
    # Execute!
    # This might take memory. 1.5GB log -> ~11M lines.
    # Polars should handle it.
    
    logger.info("Executing Pipeline (this may take a few minutes)...")
    sessions = sessions_lazy.collect(streaming=True)
    
    logger.info(f"Total Sessions: {len(sessions)}")
    
    # 7. Labels
    label_path = Path("D:/ProLog/data/HDFS/anomaly_label.csv")
    
    if label_path.exists() and label_path.stat().st_size > 1000: # Check if big enough
        logger.info(f"Loading labels from {label_path}...")
        labels = pl.read_csv(str(label_path))
        labels = labels.cast({"BlockId": pl.Utf8})
        
        dataset = sessions.join(labels, on="BlockId", how="left")
        dataset = dataset.with_columns(
            pl.when(pl.col("Label") == "Anomaly").then(1).otherwise(0).alias("label")
        )
    else:
        logger.warning(f"Label file {label_path} missing or empty. Using DUMMY labels (All Normal).")
        dataset = sessions.with_columns(pl.lit(0).alias("label"))
        
    # 8. Split & Save
    # Train on Normal Only
    normal_df = dataset.filter(pl.col("label") == 0)
    anom_df = dataset.filter(pl.col("label") == 1)
    
    logger.info(f"Normal: {len(normal_df)}, Anomaly: {len(anom_df)}")
    
    # Shuffle
    normal_df = normal_df.sample(fraction=1.0, shuffle=True, seed=config.SEED)
    
    n_train = int(len(normal_df) * config.TRAIN_SIZE)
    train_df = normal_df.head(n_train)
    test_normal = normal_df.tail(len(normal_df) - n_train)
    test_df = pl.concat([test_normal, anom_df])
    
    # Join EventIds to string
    train_df = train_df.with_columns(pl.col("EventId").list.join(" ").alias("EventTemplate"))
    test_df = test_df.with_columns(pl.col("EventId").list.join(" ").alias("EventTemplate"))
    
    train_out = config.DATA_DIR / "HDFS_train.csv"
    test_out = config.DATA_DIR / "HDFS_test.csv"
    
    train_df.select(["label", "BlockId", "EventTemplate"]).write_csv(str(train_out))
    test_df.select(["label", "BlockId", "EventTemplate"]).write_csv(str(test_out))
    
    logger.info("Done. Files saved.")

if __name__ == "__main__":
    process_dataset()
