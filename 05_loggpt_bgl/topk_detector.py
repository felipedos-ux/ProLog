"""
Top-K Anomaly Detector para BGL

Implementa a metodologia do LogGPT:
- Predição Top-K de próximos templates
- K = 50% do vocabulário de templates
- Sequência é anômala se algum template real ∉ Top-K predito
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np

class TopKAnomalyDetector:
    """Detector de anomalias usando Top-K prediction"""
    
    def __init__(self, model, vocab_path: str, device="cpu"):
        """
        Args:
            model: LogGPT model treinado
            vocab_path: Caminho para bgl_template_vocab.json
            device: "cpu" ou "cuda"
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Carregar vocabulário
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.template_to_id = vocab_data['template_to_id']
        # Converter keys para int no id_to_template
        self.id_to_template = {int(k): v for k, v in vocab_data['id_to_template'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.num_real_templates = vocab_data['num_real_templates']
        
        # K = 50% dos templates reais
        self.K = self.num_real_templates // 2
        
        print(f"TopKAnomalyDetector initialized:")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Real templates: {self.num_real_templates}")
        print(f"  K (Top-K): {self.K}")
    
    def templates_to_ids(self, templates: List[str]) -> List[int]:
        """Converte lista de templates para IDs"""
        ids = []
        for template in templates:
            if template in self.template_to_id:
                ids.append(self.template_to_id[template])
            else:
                # Template desconhecido → usar <UNK>
                ids.append(self.template_to_id["<UNK>"])
        return ids
    
    def predict_top_k(self, context_ids: List[int], k: int = None) -> List[int]:
        """
        Prediz Top-K próximos templates dado um contexto
        
        Args:
            context_ids: Lista de IDs de templates anteriores
            k: Número de top predictions (default: self.K)
        
        Returns:
            Lista dos K template IDs mais prováveis
        """
        if k is None:
            k = self.K
        
        # Preparar input
        input_tensor = torch.tensor([context_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            logits, _ = self.model(input_tensor)
            
            # Logits da última posição (próximo token)
            next_token_logits = logits[0, -1, :]
            
            # Aplicar softmax
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Top-K
            top_k_probs, top_k_ids = torch.topk(probs, k)
            
            return top_k_ids.cpu().tolist()
    
    def detect_anomaly(self, templates: List[str]) -> Tuple[bool, dict]:
        """
        Detecta se sequência de templates é anômala
        
        Metodologia LogGPT:
        - Para cada posição i (a partir de i=1):
          - Prediz Top-K próximos templates dado contexto templates[:i]
          - Se templates[i] ∉ Top-K → ANOMALIA
        
        Args:
            templates: Lista de templates da sequência
        
        Returns:
            (is_anomaly, details) onde details contém informações de debug
        """
        if len(templates) < 2:
            # Sequência muito curta, considerar normal
            return False, {"reason": "sequence_too_short"}
        
        # Converter para IDs
        template_ids = self.templates_to_ids(templates)
        
        # Para cada posição (exceto a primeira)
        anomaly_positions = []
        predictions_log = []
        
        for i in range(1, len(template_ids)):
            context = template_ids[:i]
            actual_id = template_ids[i]
            
            # Predizer Top-K
            top_k_ids = self.predict_top_k(context, k=self.K)
            
            # Verificar se o template real está no Top-K
            if actual_id not in top_k_ids:
                anomaly_positions.append(i)
            
            predictions_log.append({
                'position': i,
                'actual': self.id_to_template.get(actual_id, "<UNK>"),
                'actual_id': actual_id,
                'in_top_k': actual_id in top_k_ids,
                'top_k_ids': top_k_ids[:5]  # Salvar apenas top-5 para debug
            })
        
        is_anomaly = len(anomaly_positions) > 0
        
        details = {
            'anomaly_positions': anomaly_positions,
            'num_anomaly_positions': len(anomaly_positions),
            'sequence_length': len(templates),
            'anomaly_ratio': len(anomaly_positions) / (len(templates) - 1) if len(templates) > 1 else 0,
            'predictions_log': predictions_log
        }
        
        return is_anomaly, details
    
    def evaluate(self, test_sequences: List[Tuple[List[str], int]]) -> dict:
        """
        Avalia detector em um conjunto de teste
        
        Args:
            test_sequences: Lista de (templates, label) onde:
                - templates: Lista de templates da sequência
                - label: 0 (normal) ou 1 (anomalia)
        
        Returns:
            Dict com métricas: precision, recall, f1, confusion matrix
        """
        print(f"\n🔍 Evaluating on {len(test_sequences)} sequences...")
        
        y_true = []
        y_pred = []
        
        for i, (templates, label) in enumerate(test_sequences):
            is_anomaly, _ = self.detect_anomaly(templates)
            
            y_true.append(label)
            y_pred.append(1 if is_anomaly else 0)
            
            if i % 100 == 0:
                print(f"   Processed {i}/{len(test_sequences)}")
        
        # Calcular métricas
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'accuracy': (tp + tn) / len(y_true),
            'total_sequences': len(test_sequences)
        }
        
        return results
