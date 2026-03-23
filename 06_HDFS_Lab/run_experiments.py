"""
Experiment Orchestrator: Runs all HDFS Lab experiments sequentially.
Each experiment trains a model, runs detection, and saves results.

Usage:
    python run_experiments.py                  # Run ALL experiments
    python run_experiments.py --experiment A   # Run single experiment
    python run_experiments.py --experiment B C  # Run specific experiments
    python run_experiments.py --skip-data       # Skip data sampling (already done)
"""
import argparse
import json
import time
import os
import sys
import torch
from pathlib import Path
from datetime import datetime

from config import (
    EXPERIMENTS, RESULTS_DIR, MODELS_DIR, LAB_DATA_DIR,
    DEVICE, TOP_K_RATIO, set_seeds
)
from utils.logger import setup_logger

logger = setup_logger('orchestrator')


def ensure_data_exists():
    """Check if subset data exists, create if not."""
    train_file = LAB_DATA_DIR / "hdfs_train_5k.csv"
    test_file = LAB_DATA_DIR / "hdfs_test_subset.csv"
    
    if train_file.exists() and test_file.exists():
        logger.info("✅ Subset data already exists, skipping sampling.")
        return True
    
    logger.info("🔄 Creating data subsets...")
    from data_sampler import create_subsets
    create_subsets()
    return True


def run_single_experiment(exp_id: str) -> dict:
    """
    Run a single experiment end-to-end.
    
    Args:
        exp_id: 'A', 'B', 'C', 'D', 'E', or 'F'
    
    Returns:
        result: dict with all experiment data
    """
    exp_config = EXPERIMENTS[exp_id]
    
    logger.info("\n" + "=" * 70)
    logger.info(f"🧪 EXPERIMENT {exp_id}: {exp_config['name']}")
    logger.info(f"   {exp_config['description']}")
    logger.info("=" * 70)
    
    set_seeds()
    t0 = time.time()
    
    result = {
        'experiment_id': exp_id,
        'name': exp_config['name'],
        'description': exp_config['description'],
        'config': {
            'arch': exp_config['arch'],
            'detection': exp_config['detection'],
            'deduplicate': exp_config['deduplicate'],
            'use_rl': exp_config['use_rl'],
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    # ============================
    # PHASE 1: TRAINING
    # ============================
    from train import train_model
    from model import LogGPT
    import torch
    
    model_dir = MODELS_DIR / f"exp_{exp_id.lower()}"
    model_path = model_dir / "model.pt"
    config_path = model_dir / "config.pt"
    meta_path = model_dir / "train_meta.json"
    
    if model_path.exists() and config_path.exists() and meta_path.exists():
        logger.info(f"⏭️  Found existing pre-trained model. Skipping Phase 1.")
        gpt_config = torch.load(config_path, map_location=DEVICE, weights_only=False)
        model = LogGPT(gpt_config)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        model.to(DEVICE)
        with open(meta_path, 'r') as f:
            train_meta = json.load(f)
    else:
        model, gpt_config, train_meta = train_model(
            experiment_id=exp_id,
            arch_config=exp_config['arch'],
            deduplicate=exp_config['deduplicate'],
            is_regex=exp_config.get('is_regex', False),
        )
    
    result['training'] = {
        'phase': 'pretraining',
        'epochs_run': train_meta.get('epochs_run', 0),
        'best_val_loss': train_meta.get('best_val_loss', 0),
        'train_time_seconds': train_meta.get('train_time_seconds', 0),
        'param_count_M': train_meta.get('param_count_M', 0),
        'data_meta': train_meta.get('data_meta', {}),
    }
    
    # ============================
    # PHASE 2: RL (if applicable)
    # ============================
    if exp_config['use_rl']:
        rl_model_path = model_dir / "model_rl.pt"
        rl_meta_path = model_dir / "rl_meta.json"
        
        if rl_model_path.exists() and rl_meta_path.exists():
            logger.info(f"⏭️  Found existing RL-finetuned model. Skipping Phase 2.")
            model.load_state_dict(torch.load(rl_model_path, map_location=DEVICE, weights_only=False))
            with open(rl_meta_path, 'r') as f:
                rl_meta = json.load(f)
        else:
            from detect_topk import calculate_dynamic_k
            k, _ = calculate_dynamic_k()
            
            from rl_trainer import run_rl_finetuning
            model, rl_meta = run_rl_finetuning(
                experiment_id=exp_id,
                model=model,
                config=gpt_config,
                k_top=10, # default to 10 for RL to not be overly strict
                deduplicate=exp_config['deduplicate'],
            )
            
        result['rl_finetuning'] = {
            'k_top': rl_meta.get('k_top', 0),
            'episodes_run': rl_meta.get('episodes_run', 0),
            'rl_time_seconds': rl_meta.get('rl_time_seconds', 0),
            'final_avg_reward': rl_meta.get('final_avg_reward', 0),
        }
    
    # ============================
    # PHASE 3: DETECTION
    # ============================
    if exp_config['detection'] == 'topk':
        from detect_topk import run_topk_detection
        metrics, det_results = run_topk_detection(
            model=model,
            config=gpt_config,
            experiment_id=exp_id,
        )
    else:
        from detect_threshold import run_threshold_detection
        metrics, det_results = run_threshold_detection(
            model=model,
            config=gpt_config,
            experiment_id=exp_id,
            is_regex=exp_config.get('is_regex', False),
            k_sigma=exp_config.get('k_sigma', 2.0),
        )
    
    result['metrics'] = metrics
    
    # Total time
    total_time = time.time() - t0
    result['total_time_seconds'] = round(total_time, 1)
    
    # Save result
    result_file = RESULTS_DIR / f"exp_{exp_id.lower()}_{exp_config['name'].lower().replace(' ', '_')}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"\n✅ Experiment {exp_id} complete in {total_time:.0f}s")
    logger.info(f"   F1={metrics['f1']:.4f} | P={metrics['precision']:.4f} | R={metrics['recall']:.4f}")
    logger.info(f"   Saved to {result_file}")
    
    return result


def print_comparison_table(results: dict):
    """Print a nice comparison table of all experiment results."""
    
    logger.info("\n" + "=" * 90)
    logger.info("📊 EXPERIMENT COMPARISON TABLE")
    logger.info("=" * 90)
    
    # Header
    header = f"{'Exp':>3} | {'Name':<20} | {'F1':>7} | {'Prec':>7} | {'Recall':>7} | {'TP':>4} | {'FP':>4} | {'FN':>4} | {'Time':>6} | {'Detection':<10}"
    logger.info(header)
    logger.info("-" * 90)
    
    # Sort by experiment ID
    for exp_id in sorted(results.keys()):
        r = results[exp_id]
        m = r.get('metrics', {})
        name = r.get('name', '')[:20]
        f1 = m.get('f1', 0)
        prec = m.get('precision', 0)
        recall = m.get('recall', 0)
        tp = m.get('tp', 0)
        fp = m.get('fp', 0)
        fn = m.get('fn', 0)
        total_time = r.get('total_time_seconds', 0)
        detection = m.get('detection_method', '?')
        
        row = f"  {exp_id} | {name:<20} | {f1:>7.4f} | {prec:>7.4f} | {recall:>7.4f} | {tp:>4} | {fp:>4} | {fn:>4} | {total_time:>5.0f}s | {detection:<10}"
        logger.info(row)
    
    logger.info("-" * 90)
    
    # Find best
    best_id = max(results.keys(), key=lambda k: results[k].get('metrics', {}).get('f1', 0))
    best_f1 = results[best_id].get('metrics', {}).get('f1', 0)
    
    logger.info(f"\n🏆 Best: Experiment {best_id} ({results[best_id]['name']}) with F1={best_f1:.4f}")
    
    # Baseline comparison
    if 'A' in results:
        baseline_f1 = results['A'].get('metrics', {}).get('f1', 0)
        logger.info(f"\n📈 Improvements vs Baseline (A):")
        for exp_id in sorted(results.keys()):
            if exp_id == 'A':
                continue
            exp_f1 = results[exp_id].get('metrics', {}).get('f1', 0)
            diff = exp_f1 - baseline_f1
            sign = "+" if diff >= 0 else ""
            logger.info(f"   {exp_id} ({results[exp_id]['name']}): {sign}{diff:.4f} ({sign}{diff*100:.2f}pp)")
    
    # Lead Time comparison
    logger.info(f"\n⏱️ Lead Time Analysis:")
    for exp_id in sorted(results.keys()):
        r = results[exp_id]
        lt = r.get('metrics', {}).get('lead_time', {})
        if lt:
            ant = lt.get('anticipated_count', 0)
            avg = lt.get('avg_minutes', 0)
            logger.info(f"   {exp_id}: {ant} anticipated, avg={avg:.1f}min")
    
    logger.info("=" * 90)


def save_final_summary(results: dict):
    """Save the final comparison summary as JSON."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'experiments': {}
    }
    
    for exp_id, r in sorted(results.items()):
        summary['experiments'][exp_id] = {
            'name': r.get('name', ''),
            'f1': r.get('metrics', {}).get('f1', 0),
            'precision': r.get('metrics', {}).get('precision', 0),
            'recall': r.get('metrics', {}).get('recall', 0),
            'detection_method': r.get('metrics', {}).get('detection_method', ''),
            'total_time_seconds': r.get('total_time_seconds', 0),
            'use_rl': r.get('config', {}).get('use_rl', False),
            'deduplicate': r.get('config', {}).get('deduplicate', False),
        }
    
    summary_file = RESULTS_DIR / "final_comparison.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📄 Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="HDFS Lab Experiment Runner")
    parser.add_argument('--experiment', '-e', nargs='+', choices=list(EXPERIMENTS.keys()),
                       help='Run specific experiment(s). Default: run all.')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data sampling step.')
    args = parser.parse_args()
    
    logger.info("🔬 HDFS Lab — Experiment Orchestrator")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Step 1: Ensure data exists
    if not args.skip_data:
        ensure_data_exists()
    
    # Step 2: Determine which experiments to run
    exp_ids = args.experiment or list(EXPERIMENTS.keys())
    logger.info(f"\nExperiments to run: {', '.join(exp_ids)}")
    
    # Step 3: Run experiments
    all_results = {}
    
    for exp_id in exp_ids:
        try:
            result = run_single_experiment(exp_id)
            all_results[exp_id] = result
        except Exception as e:
            logger.error(f"❌ Experiment {exp_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_id] = {
                'experiment_id': exp_id,
                'name': EXPERIMENTS[exp_id]['name'],
                'error': str(e),
                'metrics': {'f1': 0, 'precision': 0, 'recall': 0},
            }
    
    # Step 4: Comparison
    if len(all_results) > 1:
        print_comparison_table(all_results)
    
    # Step 5: Save summary
    save_final_summary(all_results)
    
    logger.info("\n🎉 All experiments complete!")


if __name__ == '__main__':
    main()
