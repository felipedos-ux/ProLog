import sys
from pathlib import Path
import shutil
import config_hdfs as config

# Add current dir to path
# Add parent dir to path to find 'utils'
sys.path.append(str(Path.cwd().parent))

import dataset_hdfs
import train_hdfs
import detect_hdfs

# Setup Dummy Data
TEST_DIR = Path("D:/ProLog/data/HDFS_TEST")
TEST_DIR.mkdir(parents=True, exist_ok=True)

DUMMY_LOG = TEST_DIR / "HDFS.log"
DUMMY_LABEL = TEST_DIR / "anomaly_label.csv"

# Sample HDFS Logs (from HDFS_2k or valid patterns)
LOG_CONTENT = """081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.10.6:50010 is added to blk_-1608999687919862906 size 91178
081109 203519 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.10.6:40524 dest: /10.250.10.6:50010
081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating
081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-1608999687919862906 terminating
081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-1608999687919862906 terminating
081109 203520 142 INFO dfs.DataNode$DataXceiver: Receiving block blk_7503483334202473044 src: /10.250.10.6:43696 dest: /10.250.10.6:50010
081109 203615 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-3544583377289625738 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
"""
# Labels
LABEL_CONTENT = """BlockId,Label
blk_-1608999687919862906,Normal
blk_7503483334202473044,Anomaly
blk_-3544583377289625738,Normal
"""

def setup():
    print("Setting up Dummy Data...")
    with open(DUMMY_LOG, 'w') as f:
        f.write(LOG_CONTENT)
    with open(DUMMY_LABEL, 'w') as f:
        f.write(LABEL_CONTENT)
    
    # Override Config
    config.DATA_DIR = TEST_DIR
    config.HDFS_DIR = TEST_DIR
    config.LOG_FILE = DUMMY_LOG
    config.LABEL_FILE = DUMMY_LABEL
    config.MODEL_DIR = TEST_DIR / "models"
    config.MODEL_DIR.mkdir(exist_ok=True)
    
    config.EPOCHS = 1
    config.BATCH_SIZE = 2
    config.TRAIN_SIZE = 0.5 # Force split
    
    print("Config Updated.")

def run_pipeline():
    setup()
    
    print("\n--- Running Dataset Preprocessing ---")
    dataset_hdfs.process_dataset()
    
    print("\n--- Running Training ---")
    train_hdfs.main()
    
    print("\n--- Running Detection ---")
    detect_hdfs.detect()
    
    print("\n--- Verification ---")
    res_file = config.DATA_DIR / "HDFS_test_results.csv"
    if res_file.exists():
        print("SUCCESS: Results file generated.")
        with open(res_file, 'r') as f:
            print(f.read())
    else:
        print("FAILURE: Results file not found.")

if __name__ == "__main__":
    run_pipeline()
