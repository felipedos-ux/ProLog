# Plano de Melhoria: LogGPT-Small

**Data**: 06/02/2026  
**Baseado em**: `VALIDACAO_PROLOG_COMPLETA.md`

---

## 📋 Resumo dos Problemas

| # | Problema | Prioridade | Status |
|---|----------|------------|--------|
| 1 | Threshold hardcoded (5.0) | 🔴 CRÍTICO | ⏳ Pendente |
| 2 | Validation set não utilizado | 🔴 CRÍTICO | ⏳ Pendente |
| 3 | Loss values idênticos (17.43) | 🔴 CRÍTICO | ⏳ Pendente |
| 4 | Sem validação durante treino | 🟡 ALTO | ⏳ Pendente |
| 5 | Sem early stopping | 🟡 ALTO | ⏳ Pendente |
| 6 | Separação de logs fraca | 🟢 MÉDIO | ⏳ Pendente |
| 7 | Hiperparâmetros não documentados | 🟢 MÉDIO | ⏳ Pendente |
| 8 | Métricas incompletas (sem AUC) | 🟢 MÉDIO | ⏳ Pendente |

---

## 🔴 Correções CRÍTICAS

### 1. Calibrar Threshold no Validation Set

**Problema**: Threshold 5.0 hardcoded sem calibração formal.

**Solução**: Criar `calibrate_threshold.py`

```python
# Pseudocódigo
for threshold in np.arange(1.0, 20.0, 0.5):
    metrics = evaluate_on_val_set(model, val_ids, threshold)
    if metrics['f1'] > best_f1:
        best_threshold = threshold

# Salvar threshold ótimo
save_threshold(best_threshold)  # → optimal_threshold.txt
```

**Entregáveis**:
- [ ] `calibrate_threshold.py`
- [ ] `optimal_threshold.txt`
- [ ] Curva ROC (imagem)
- [ ] Curva Precision-Recall (imagem)

---

### 2. Usar Validation Set

**Problema**: `val_ids` calculado mas nunca usado.

**Solução**: Modificar `detect_custom.py`

```python
# ANTES
val_ids, test_norm_ids = train_test_split(...)  # val_ids ignorado

# DEPOIS
# 1. Calibrar no val_ids
threshold = calibrate_threshold(model, val_ids)

# 2. Avaliar no test_norm_ids + anom_ids
results = evaluate(model, test_ids, threshold)
```

---

### 3. Investigar Loss Idêntico

**Problema**: Todos 169 casos têm loss = 17.43 (estatisticamente impossível).

**Diagnóstico Necessário**:

```python
# Adicionar logging detalhado
for i, log in enumerate(session_logs):
    loss = calculate_loss(log)
    print(f"Log {i}: loss={loss:.6f}, tokens={len(new_ids)}")
    
    all_losses.append(loss)

# Plotar distribuição
plt.hist(all_losses, bins=50)
plt.savefig("loss_distribution.png")
```

**Hipóteses**:
1. Bug no cálculo de `logit_indices`
2. Bug no `target_start_idx`
3. Modelo degenerado (sempre prediz mesmo token)

---

## 🟡 Correções ALTAS

### 4. Validação Durante Treino

**Problema**: Sem val_loss, impossível detectar overfitting.

**Solução**: Modificar `train_custom.py`

```python
for epoch in range(EPOCHS):
    # Treino
    train_loss = train_epoch(model, train_loader)
    
    # Validação (NOVO)
    val_loss = evaluate_epoch(model, val_loader)
    
    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    # Salvar curva
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plotar learning curve
plot_learning_curve(train_losses, val_losses)
```

---

### 5. Early Stopping

**Problema**: 10 epochs fixo sem justificativa.

**Solução**:

```python
patience = 3
best_val_loss = float('inf')
counter = 0

for epoch in range(MAX_EPOCHS):
    val_loss = evaluate(model, val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

## 🟢 Correções MÉDIAS

### 6. Melhorar Separação de Logs

**Atual**: `" \n ".join(logs)`

**Proposta**: Usar token especial `<|LOG|>`

```python
# dataset.py
LOG_SEPARATOR = " <|LOG|> "
session_text = LOG_SEPARATOR.join(logs)
```

---

### 7. Documentar Hiperparâmetros

Adicionar ao README:

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| block_size | 128 | Média de tokens por sessão |
| epochs | 10 | Convergência observada |
| batch_size | 32 | Limite GPU |
| learning_rate | 5e-4 | Padrão AdamW |

---

### 8. Adicionar Métricas

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# Coletar scores contínuos
y_true = [...]  # 0/1 labels
y_scores = [...]  # max_loss por sessão

# Calcular
auc_roc = roc_auc_score(y_true, y_scores)
auc_pr = average_precision_score(y_true, y_scores)
```

---

## 📅 Cronograma

| Semana | Tarefas | Esforço |
|--------|---------|---------|
| **1** | Problemas 1-3 (Críticos) | 8h |
| **2** | Problemas 4-5 (Altos) + Retreino | 6h |
| **3** | Problemas 6-8 (Médios) | 4h |
| **4** | Documentação + Revisão Final | 4h |

**Total**: ~22 horas

---

## ✅ Checklist de Validação

### Antes de Publicação

- [ ] Threshold calibrado no validation set
- [ ] Bug de loss investigado e corrigido
- [ ] Validação durante treino implementada
- [ ] Early stopping implementado
- [ ] Curva de aprendizado plotada
- [ ] Curva ROC incluída
- [ ] Precision-Recall curve incluída
- [ ] README atualizado com hiperparâmetros
- [ ] Resultados revalidados no test set

---

## 📚 Referências

1. Fawcett (2006) - "An introduction to ROC analysis"
2. Prechelt (1998) - "Early Stopping - But When?"
3. Kaufman et al. (2012) - "Leakage in Data Mining"
