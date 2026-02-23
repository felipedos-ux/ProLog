# -*- coding: utf-8 -*-
"""
HDFS Full Pipeline: Train -> Calibrate -> Detect
Executes the complete pipeline from scratch in the external terminal.
Optimized for RTX 3080 Ti (12GB) + Ryzen 3600 (12 threads) + 32GB RAM.

Usage:
    cd D:\\ProLog\\03_HDFS_Benchmark
    python run_full_pipeline.py
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta


WORK_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(WORK_DIR, "saved_models")

CLEANUP_FILES = [
    os.path.join(SAVED_MODELS_DIR, "hdfs_loggpt.pt"),
    os.path.join(SAVED_MODELS_DIR, "config.pt"),
    os.path.join(WORK_DIR, "threshold_config.json"),
    os.path.join(WORK_DIR, "results_chunked.txt"),
    os.path.join(WORK_DIR, "calibration_checkpoint.pkl"),
    os.path.join(WORK_DIR, "detection_checkpoint.pkl"),
    os.path.join(WORK_DIR, "detection_results_partial.pkl"),
]


def print_header(title):
    print()
    print("=" * 60)
    print("  " + title)
    print("=" * 60)
    print()


def print_step(step, total, desc):
    line = "-" * 60
    print("\n" + line)
    print("  Step {}/{}: {}".format(step, total, desc))
    print(line + "\n")


def cleanup():
    print_header("LIMPEZA: Removendo arquivos anteriores")
    removed = 0
    for f in CLEANUP_FILES:
        if os.path.exists(f):
            os.remove(f)
            print("  [OK] Removido: " + os.path.basename(f))
            removed += 1
        else:
            print("  [--] Nao existe: " + os.path.basename(f))
    print("\n  Total removidos: {} arquivos".format(removed))
    return removed


def run_script(script_name, description):
    script_path = os.path.join(WORK_DIR, script_name)
    if not os.path.exists(script_path):
        print("  [ERRO] Script nao encontrado: " + script_path)
        return False, 0

    start = time.time()
    print("  >>> Executando: python " + script_name)
    print("  >>> Inicio: " + datetime.now().strftime("%H:%M:%S"))
    print()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=WORK_DIR,
    )

    elapsed = time.time() - start
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    print()
    if result.returncode == 0:
        print("  [OK] {} concluido com sucesso!".format(description))
        print("  [OK] Tempo: " + elapsed_str)
    else:
        print("  [ERRO] {} falhou! (exit code: {})".format(description, result.returncode))
        print("  [OK] Tempo: " + elapsed_str)

    return result.returncode == 0, elapsed


def verify_training():
    model_path = os.path.join(SAVED_MODELS_DIR, "hdfs_loggpt.pt")
    config_path = os.path.join(SAVED_MODELS_DIR, "config.pt")

    if not os.path.exists(model_path):
        print("  [ERRO] Modelo nao encontrado: hdfs_loggpt.pt")
        return False
    if not os.path.exists(config_path):
        print("  [ERRO] Config nao encontrado: config.pt")
        return False

    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print("  [OK] Modelo salvo: hdfs_loggpt.pt ({:.1f} MB)".format(model_size))
    print("  [OK] Config salvo: config.pt")
    return True


def verify_calibration():
    config_path = os.path.join(WORK_DIR, "threshold_config.json")
    if not os.path.exists(config_path):
        print("  [ERRO] threshold_config.json nao encontrado!")
        return False

    import json
    with open(config_path, "r") as f:
        config = json.load(f)

    print("  [OK] Threshold: {:.4f}".format(config.get("threshold", 0)))
    print("  [OK] F1 Score:  {:.4f}".format(config.get("f1_score", 0)))
    print("  [OK] Precision: {:.4f}".format(config.get("precision", 0)))
    print("  [OK] Recall:    {:.4f}".format(config.get("recall", 0)))
    print("  [OK] Method:    {}".format(config.get("method", "N/A")))
    return True


def verify_detection():
    results_path = os.path.join(WORK_DIR, "results_chunked.txt")
    if not os.path.exists(results_path):
        print("  [ERRO] results_chunked.txt nao encontrado!")
        return False

    with open(results_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("  [OK] Resultados gerados:\n")
    print(content)
    return True


def main():
    pipeline_start = time.time()

    print_header("HDFS PIPELINE COMPLETO (Do Zero)")
    print("  Hardware: RTX 3080 Ti + Ryzen 3600 + 32GB RAM")
    print("  Config:   LR=1e-4, Epochs=30, Patience=5, Batch=64")
    print("  Inicio:   " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    print("  Etapas:")
    print("    1. Limpeza (remover modelo antigo)")
    print("    2. Treinamento (~2-5h com early stopping)")
    print("    3. Calibracao 100% (~1.5h)")
    print("    4. Deteccao 100% (~30-40min)")
    print()

    times = {}

    # STEP 1: Cleanup
    print_step(1, 4, "LIMPEZA")
    cleanup()

    # STEP 2: Training
    print_step(2, 4, "TREINAMENTO (LR=1e-4, Epochs=30, Patience=5)")
    success, elapsed = run_script("train.py", "Treinamento")
    times["train"] = elapsed

    if not success:
        print("\n  [ERRO] TREINAMENTO FALHOU! Pipeline interrompido.")
        return 1

    print("\n  Verificando resultado do treinamento...")
    if not verify_training():
        print("\n  [ERRO] Verificacao falhou! Pipeline interrompido.")
        return 1

    # STEP 3: Calibration
    print_step(3, 4, "CALIBRACAO (100% sessoes)")
    success, elapsed = run_script("calibrate_optimized.py", "Calibracao")
    times["calibrate"] = elapsed

    if not success:
        print("\n  [ERRO] CALIBRACAO FALHOU! Pipeline interrompido.")
        return 1

    print("\n  Verificando resultado da calibracao...")
    if not verify_calibration():
        print("\n  [ERRO] Verificacao falhou! Pipeline interrompido.")
        return 1

    # STEP 4: Detection
    print_step(4, 4, "DETECCAO (100% sessoes)")
    success, elapsed = run_script("detect_chunked.py", "Deteccao")
    times["detect"] = elapsed

    if not success:
        print("\n  [ERRO] DETECCAO FALHOU! Pipeline interrompido.")
        return 1

    print("\n  Verificando resultado da deteccao...")
    if not verify_detection():
        print("\n  [ERRO] Verificacao falhou!")

    # FINAL SUMMARY
    pipeline_elapsed = time.time() - pipeline_start

    print_header("PIPELINE COMPLETO!")
    print("  Tempo Total: " + str(timedelta(seconds=int(pipeline_elapsed))))
    print()
    print("  Tempos por Etapa:")
    for step, elapsed in times.items():
        print("    {:>12}: {}".format(step, str(timedelta(seconds=int(elapsed)))))
    print()
    print("  Arquivos Gerados:")
    print("    - saved_models/hdfs_loggpt.pt  (modelo treinado)")
    print("    - saved_models/config.pt       (configuracao)")
    print("    - threshold_config.json        (threshold calibrado)")
    print("    - results_chunked.txt          (resultados deteccao)")
    print()
    print("  Concluido: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
