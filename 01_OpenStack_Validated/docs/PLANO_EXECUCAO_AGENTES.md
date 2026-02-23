# Plano de Execução: LogGPT-Small com Agentes Especializados

**Data**: 06/02/2026  
**Baseado em**: `PLANO_MELHORIA.md`

---

## 🤖 Agentes Selecionados (MCP ai-context)

| Agente | Especialidade | Responsabilidades |
|--------|---------------|-------------------|
| **bug-fixer** | Correção de bugs | Investigar loss idêntico, corrigir cálculos |
| **feature-developer** | Novas features | Calibração de threshold, early stopping |
| **backend-specialist** | Lógica de servidor | Validação durante treino, métricas |
| **documentation-writer** | Documentação | README, hiperparâmetros, relatórios |
| **devops-specialist** | Deploy/CI | Testes, reprodutibilidade, seeds |

---

## 📋 Distribuição de Tarefas

### 🔴 SEMANA 1: Correções Críticas

#### Tarefa 1.1: Investigar Bug de Loss Idêntico
**Agente**: `bug-fixer` 🐛  
**Prioridade**: CRÍTICA  
**Esforço**: 3h

**Objetivo**: Descobrir por que todos os 169 casos têm loss = 17.43

**Ações**:
1. Adicionar logging detalhado em `detect_custom.py` (linhas 126-134)
   ```python
   print(f"Log {i}: loss={loss_val:.6f}, logits_shape={relevant_logits.shape}")
   ```
2. Coletar distribuição de losses (todos os logs, não apenas > threshold)
3. Plotar histograma de losses (normal vs anômalo)
4. Verificar cálculo de `logit_indices` e `target_start_idx`
5. Adicionar assertions para validar shapes

**Entregáveis**:
- [ ] `loss_distribution.png` (histograma)
- [ ] Relatório de diagnóstico (bug encontrado ou comportamento explicado)
- [ ] Correção aplicada (se bug confirmado)

---

#### Tarefa 1.2: Calibrar Threshold no Validation Set
**Agente**: `feature-developer` ⚙️  
**Prioridade**: CRÍTICA  
**Esforço**: 4h

**Objetivo**: Criar calibração formal do threshold usando val_ids

**Ações**:
1. Criar novo arquivo `calibrate_threshold.py`
2. Implementar função `calculate_session_losses(model, session_ids)`
3. Testar thresholds de 1.0 a 20.0 (step 0.1)
4. Calcular Precision, Recall, F1 para cada threshold
5. Plotar curvas ROC e Precision-Recall
6. Salvar threshold ótimo em `optimal_threshold.txt`

**Código Base**:
```python
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def calibrate_threshold(model, tokenizer, df, val_norm_ids, val_anom_ids):
    # Calcular losses
    losses = []
    labels = []
    
    for tid in val_norm_ids:
        max_loss = get_max_loss_for_session(model, tid)
        losses.append(max_loss)
        labels.append(0)
    
    for tid in val_anom_ids:
        max_loss = get_max_loss_for_session(model, tid)
        losses.append(max_loss)
        labels.append(1)
    
    # Testar thresholds
    thresholds = np.arange(1.0, 20.0, 0.1)
    best_f1 = 0
    best_threshold = None
    
    for t in thresholds:
        preds = [1 if loss > t else 0 for loss in losses]
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    return best_threshold
```

**Entregáveis**:
- [ ] `calibrate_threshold.py`
- [ ] `optimal_threshold.txt`
- [ ] `roc_curve.png`
- [ ] `precision_recall_curve.png`

---

#### Tarefa 1.3: Usar Validation Set em detect_custom.py
**Agente**: `feature-developer` ⚙️  
**Prioridade**: CRÍTICA  
**Esforço**: 1h

**Objetivo**: Modificar `detect_custom.py` para usar threshold calibrado

**Ações**:
1. Carregar threshold de `optimal_threshold.txt` em vez de hardcoded
2. Adicionar comentário explicando origem do threshold
3. Validar que val_ids não é usado na avaliação final

**Código**:
```python
# ANTES
THRESHOLD = 5.0  # Determined from findings

# DEPOIS
# Carregar threshold calibrado no validation set
with open("optimal_threshold.txt", "r") as f:
    THRESHOLD = float(f.read().strip())
print(f"   Using calibrated threshold: {THRESHOLD:.2f}")
```

**Entregáveis**:
- [ ] `detect_custom.py` atualizado

---

### 🟡 SEMANA 2: Validação e Retreino

#### Tarefa 2.1: Adicionar Validação Durante Treino
**Agente**: `backend-specialist` 🔧  
**Prioridade**: ALTA  
**Esforço**: 3h

**Objetivo**: Implementar val_loss tracking em `train_custom.py`

**Ações**:
1. Criar `val_loader` a partir de `val_ids`
2. Adicionar função `evaluate_epoch(model, val_loader)`
3. Calcular val_loss após cada epoch
4. Salvar histórico de train_loss e val_loss
5. Plotar learning curve

**Código**:
```python
# Criar validation loader
val_dataset = lm_datasets.select(val_indices)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # Treino
    model.train()
    train_loss = train_epoch(model, train_loader, optimizer)
    
    # Validação
    model.eval()
    val_loss = evaluate_epoch(model, val_loader)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

# Plotar
plot_learning_curve(train_losses, val_losses)
```

**Entregáveis**:
- [ ] `train_custom.py` atualizado
- [ ] `learning_curve.png`

---

#### Tarefa 2.2: Implementar Early Stopping
**Agente**: `feature-developer` ⚙️  
**Prioridade**: ALTA  
**Esforço**: 2h

**Objetivo**: Adicionar early stopping baseado em val_loss

**Ações**:
1. Adicionar parâmetros `patience=3` e `best_val_loss`
2. Salvar checkpoint quando val_loss melhora
3. Parar treino se não melhorar por 3 epochs
4. Carregar melhor checkpoint ao final

**Código**:
```python
patience = 3
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(MAX_EPOCHS):
    val_loss = evaluate_epoch(model, val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pt")
        patience_counter = 0
        print("   ✓ New best model saved")
    else:
        patience_counter += 1
        print(f"   ⚠ No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Carregar melhor modelo
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pt"))
```

**Entregáveis**:
- [ ] `train_custom.py` atualizado com early stopping

---

#### Tarefa 2.3: Retreinar Modelo
**Agente**: `devops-specialist` 🚀  
**Prioridade**: ALTA  
**Esforço**: 1h (+ 10min GPU)

**Objetivo**: Retreinar modelo com pipeline corrigido

**Ações**:
1. Adicionar seeds para reprodutibilidade
2. Executar `train_custom.py` com validação
3. Verificar convergência na learning curve
4. Salvar modelo final em `model_weights/`

**Código (seeds)**:
```python
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

**Entregáveis**:
- [ ] Modelo retreinado
- [ ] Log de treinamento com val_loss

---

### 🟢 SEMANA 3: Melhorias Médias

#### Tarefa 3.1: Melhorar Separação de Logs
**Agente**: `feature-developer` ⚙️  
**Prioridade**: MÉDIA  
**Esforço**: 1h

**Objetivo**: Usar token especial para separar logs

**Ações**:
1. Modificar `dataset.py` linha 65
2. Usar `" <|LOG|> "` em vez de `" \n "`
3. Retreinar modelo (opcional, para validar melhoria)

**Código**:
```python
# ANTES
session_text = " \n ".join(row[0])

# DEPOIS
LOG_SEPARATOR = " <|LOG|> "
session_text = LOG_SEPARATOR.join(row[0])
```

**Entregáveis**:
- [ ] `dataset.py` atualizado

---

#### Tarefa 3.2: Adicionar Métricas Complementares
**Agente**: `backend-specialist` 🔧  
**Prioridade**: MÉDIA  
**Esforço**: 2h

**Objetivo**: Calcular AUC-ROC e Precision-Recall AUC

**Ações**:
1. Modificar `detect_custom.py` para coletar scores contínuos
2. Calcular AUC-ROC e AUC-PR
3. Plotar curvas
4. Adicionar ao relatório final

**Código**:
```python
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Coletar scores
y_true = []
y_scores = []

for tid, label in eval_list:
    max_loss = get_max_loss_for_session(model, tid)
    y_scores.append(max_loss)
    y_true.append(label)

# Calcular métricas
auc_roc = roc_auc_score(y_true, y_scores)
auc_pr = average_precision_score(y_true, y_scores)

print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR: {auc_pr:.4f}")

# Plotar
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.plot(fpr, tpr)
plt.savefig("roc_curve_final.png")
```

**Entregáveis**:
- [ ] Métricas AUC adicionadas
- [ ] `roc_curve_final.png`

---

### 📝 SEMANA 4: Documentação

#### Tarefa 4.1: Documentar Hiperparâmetros
**Agente**: `documentation-writer` 📚  
**Prioridade**: MÉDIA  
**Esforço**: 2h

**Objetivo**: Adicionar justificativas para hiperparâmetros

**Ações**:
1. Criar seção "Hiperparâmetros" no README
2. Documentar cada parâmetro com justificativa
3. Adicionar referências (se aplicável)

**Template**:
```markdown
## ⚙️ Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| block_size | 128 | Média de tokens por sessão OpenStack (~100-150) |
| batch_size | 32 | Máximo suportado por GPU RTX 3080 Ti (12GB) |
| learning_rate | 5e-4 | Taxa padrão AdamW para modelos GPT pequenos |
| epochs | 10 (max) | Early stopping com patience=3 |
| n_layer | 4 | Balanceamento capacidade vs velocidade |
| n_embd | 256 | Suficiente para vocabulário de logs (~50k tokens) |
```

**Entregáveis**:
- [ ] README atualizado

---

#### Tarefa 4.2: Atualizar Relatório Técnico
**Agente**: `documentation-writer` 📚  
**Prioridade**: MÉDIA  
**Esforço**: 2h

**Objetivo**: Atualizar documentação com correções implementadas

**Ações**:
1. Atualizar `loggpt_relatorio_detalhado.md`
2. Adicionar seção "Correções Implementadas"
3. Incluir novas métricas (AUC-ROC, AUC-PR)
4. Atualizar learning curve

**Entregáveis**:
- [ ] `loggpt_relatorio_detalhado.md` atualizado

---

## 📊 Cronograma Consolidado

| Semana | Agente Principal | Tarefas | Horas |
|--------|------------------|---------|-------|
| **1** | bug-fixer, feature-developer | 1.1, 1.2, 1.3 | 8h |
| **2** | backend-specialist, feature-developer, devops | 2.1, 2.2, 2.3 | 6h |
| **3** | feature-developer, backend-specialist | 3.1, 3.2 | 3h |
| **4** | documentation-writer | 4.1, 4.2 | 4h |

**Total**: 21 horas

---

## ✅ Checklist Final

### Código
- [ ] Bug de loss investigado (Tarefa 1.1)
- [ ] Threshold calibrado (Tarefa 1.2)
- [ ] Validação durante treino (Tarefa 2.1)
- [ ] Early stopping (Tarefa 2.2)
- [ ] Modelo retreinado (Tarefa 2.3)
- [ ] Separador de logs melhorado (Tarefa 3.1)
- [ ] Métricas AUC adicionadas (Tarefa 3.2)

### Documentação
- [ ] Hiperparâmetros documentados (Tarefa 4.1)
- [ ] Relatório técnico atualizado (Tarefa 4.2)

### Artefatos Gerados
- [ ] `calibrate_threshold.py`
- [ ] `optimal_threshold.txt`
- [ ] `roc_curve.png`
- [ ] `precision_recall_curve.png`
- [ ] `learning_curve.png`
- [ ] `loss_distribution.png`
- [ ] `roc_curve_final.png`

---

**Próximo Passo**: Executar Tarefa 1.1 (bug-fixer) para investigar loss idêntico.
