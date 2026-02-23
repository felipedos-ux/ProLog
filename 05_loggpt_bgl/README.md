# LogGPT Anomaly Detection for BGL (Count-Based)

This project implements a state-of-the-art **LogGPT** model for anomaly detection on the BlueGene/L (BGL) dataset. It utilizes a **Count-Based Window Strategy (N=20)** to achieve balanced and robust performance metrics.

## 🚀 Key Features

*   **Count-Based Windowing:** Logs are grouped into fixed sequences of 20, mimicking session-based workflows.
*   **Top-K Detection:** Anomaly detection based on Next Token Prediction (Top-5).
*   **Hybrid Detection:** Combines LogGPT with Frequency Map to catch rare templates.
*   **Error Categorization:** Automatically classifies failures (e.g., `System/Kernel`, `I/O`, `Network`) in the output report.
*   **Lead Time Analysis:** Capable of estimating Time-to-Failure for progressive degradation scenarios.

## 🛠️ Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Performance (Validation K=5)

*   **F1 Score:** 87.26%
*   **Precision:** 99.14%
*   **Recall:** 77.93%

## 🏃 Usage

### 1. Production Inference
To detect anomalies in a new log file (CSV format with `EventTemplate` and `Node` columns):

```bash
python deploy_bgl_prod.py --input path/to/logs.csv --output anomalies.json
```

### 2. Training (Retrain)
To retrain the model from scratch:

```bash
# 1. Generate Dataset
python dataset_bgl_count.py

# 2. Train Model
python train_count.py
```

## 📂 Structure

*   `deploy_bgl_prod.py`: Main production inference script.
*   `model.py`: LogGPT Transformer implementation.
*   `train_count.py`: Training pipeline.
*   `config_count.py`: Configuration parameters.
*   `dataset_bgl_count.py`: Data preprocessing logic.
*   `utils/`: Helper functions.

## 🔗 References
*   LogGPT Paper
*   OpenStack Log Anomaly Detection Standards
