
import os
import time

def monitor():
    print("🔍 Monitorando Progresso LogGPT BGL...")
    
    weights_path = "model_weights_bgl/loggpt_weights.pt"
    config_path = "threshold_config.json"
    results_path = "results_metrics_detailed.txt"
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"🕒 {time.strftime('%H:%M:%S')}")
        
        # 1. Check Weights
        if os.path.exists(weights_path):
            t = time.ctime(os.path.getmtime(weights_path))
            print(f"✅ Treino: Pesos salvos em {t}")
        else:
            print("⏳ Treino: Em andamento (aguardando loggpt_weights.pt)...")
            
        # 2. Check Calibration
        if os.path.exists(config_path):
            t = time.ctime(os.path.getmtime(config_path))
            print(f"✅ Calibração: Config salva em {t}")
        else:
            print("⏳ Calibração: Pendente...")
            
        # 3. Check Results
        if os.path.exists(results_path):
            print(f"\n🎉 PROCESSO CONCLUÍDO! Resultados em {results_path}")
            print("\nPreview:")
            with open(results_path, 'r', encoding='utf-8') as f:
                print(f.read())
            break
        
        time.sleep(10)

if __name__ == "__main__":
    monitor()
