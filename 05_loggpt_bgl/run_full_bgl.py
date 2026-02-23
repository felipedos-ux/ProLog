
import subprocess
import sys
import os
import time

def run_step(script_name, description):
    print(f"\n{'='*50}")
    print(f"🚀 STARTING: {description} ({script_name})")
    print(f"{'='*50}")
    
    start_time = time.time()
    try:
        # Run script and stream output
        result = subprocess.run([sys.executable, script_name], check=True)
        
        duration = (time.time() - start_time) / 60
        print(f"\n✅ FINISHED: {description} in {duration:.2f} minutes.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description} with exit code {e.returncode}")
        return False

def main():
    print("🎬 Starting Full BGL LogGPT Pipeline (05_loggpt_bgl)")
    
    # 1. Train
    if not run_step("train_custom.py", "Model Training (10 Epochs)"):
        return
        
    # 2. Calibrate
    if not run_step("calibrate_adaptive.py", "Threshold Calibration"):
        return
        
    # 3. Detect
    if not run_step("detect_custom.py", "Anomaly Detection & Evaluation"):
        return
        
    print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    
    if os.path.exists("results_metrics_detailed.txt"):
        with open("results_metrics_detailed.txt", "r", encoding="utf-8") as f:
            print("\n📄 FINAL REPORT PREVIEW:\n")
            print(f.read())

if __name__ == "__main__":
    main()
