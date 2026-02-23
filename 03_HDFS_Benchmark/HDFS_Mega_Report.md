# 📊 HDFS Anomaly Detection — Mega Relatório

> **Dataset**: HDFS (11.17M linhas, 72,661 sessões)  
> **Modelo**: LogGPT-Small (28.98M parâmetros)  
> **Threshold**: 0.2863 (8.0σ adaptive)  
> **Data**: 2026-02-12

---

## 1. Métricas Globais

| Métrica | Valor |
|---------|-------|
| **Precision** | **0.9498** |
| **Recall** | **0.8228** |
| **F1 Score** | **0.8818** |
| **Accuracy** | **0.9489** |
| **Specificity** | **0.9869** |

| Classe | Total | Resultado |
|--------|-------|-----------|
| **TP** (anomalia detectada) | 13,855 | ✅ |
| **TN** (normal confirmado) | 55,090 | ✅ |
| **FP** (falso alarme) | 733 | ⚠️ |
| **FN** (anomalia perdida) | 2,983 | ⚠️ |

---

## 2. Erros por Categoria

O dataset HDFS possui **29 templates únicos** agrupados em **10 categorias** de erro:

| Categoria | Total | Detectados | Perdidos | Recall | Avg Lead | Median Lead | Min Lead | Max Lead |
|-----------|-------|-----------|----------|--------|----------|-------------|----------|----------|
| **Other Exception** | 10,523 | 10,500 | 23 | **99.8%** | 161.5 min | 16.5 min | 0.0 min | 898.0 min |
| **InterruptedIOException** | 4,928 | 3,279 | 1,649 | **66.5%** | 55.2 min | 4.3 min | 0.0 min | 574.9 min |
| **NameSystem/BlockMap** | 1,307 | 0 | 1,307 | **0.0%** | — | — | — | — |
| **SocketTimeoutException** | 67 | 66 | 1 | **98.5%** | 223.1 min | 5.3 min | 0.2 min | 622.2 min |
| **EOFException** | 13 | 10 | 3 | **76.9%** | 176.4 min | 248.0 min | 58.0 min | 258.0 min |

### Insights por Categoria

> [!IMPORTANT]
> **NameSystem/BlockMap** (1,307 sessões) tem **recall 0%** — o modelo não consegue distinguir esses padrões dos normais. São sessões com apenas ~2 logs, insuficientes para gerar perplexidade alta.

> [!NOTE]
> **Other Exception** (10,523 sessões) domina o dataset e tem **99.8% recall** — praticamente perfeito. Inclui writeBlock errors, Connection Reset, e exceptions genéricas.

> [!TIP]
> **SocketTimeoutException** tem o **maior lead time médio** (223 min ≈ 3.7h), indicando que timeouts antecedem a falha com bastante antecedência.

---

## 3. Análise de Lead Time

> Lead Time = tempo entre a **primeira detecção** do modelo e o **último log** (falha) da sessão.

### Estatísticas Globais (N=7,345 sessões com lead > 0)

| Estatística | Valor |
|-------------|-------|
| **Média** | 161.22 min (2.7h) |
| **Mediana** | 16.08 min |
| **Desvio Padrão** | 234.97 min |
| **Mínimo** | 0.02 min (1.2 seg) |
| **Máximo** | 898.03 min (15.0h) |

### Distribuição do Lead Time

```
    0-1 min:  1,232 (16.8%) ████████
    1-5 min:  1,490 (20.3%) ██████████
   5-15 min:    924 (12.6%) ██████
  15-30 min:    315 ( 4.3%) ██
  30-60 min:    324 ( 4.4%) ██
 60-120 min:    442 ( 6.0%) ███
120-300 min:    574 ( 7.8%) ███
300-600 min:  1,269 (17.3%) ████████
   >600 min:    775 (10.6%) █████
```

> [!NOTE]
> **37.1%** das detecções ocorrem nos primeiros 5 minutos (detecção rápida).  
> **27.9%** ocorrem com mais de 5 horas de antecedência (alta previsibilidade).

---

## 4. Top 20 — Detecção Mais Rápida (Menor Lead Time)

| # | Lead Time | Alert Loss | Categoria | Logs |
|---|-----------|-----------|-----------|------|
| 1 | **0.02 min** | 5.8685 | Other Exception | 16 |
| 2 | 0.02 min | 5.8867 | Other Exception | 16 |
| 3 | 0.02 min | 5.8867 | Other Exception | 16 |
| 4 | 0.02 min | 5.8867 | Other Exception | 16 |
| 5 | 0.02 min | 0.4858 | Other Exception | 26 |
| 6 | 0.02 min | 0.3513 | Other Exception | 16 |
| 7 | 0.02 min | 0.5329 | Other Exception | 20 |
| 8 | 0.02 min | 5.8867 | Other Exception | 16 |
| 9 | 0.02 min | 5.8685 | Other Exception | 16 |
| 10 | 0.02 min | 0.5329 | Other Exception | 20 |

> Sessões com loss alto (5.88) são detectadas no **primeiro log** — padrão altamente anômalo desde o início.

---

## 5. Top 20 — Maior Antecipação (Maior Lead Time)

| # | Lead Time | Alert Loss | Categoria | Logs |
|---|-----------|-----------|-----------|------|
| 1 | **898.03 min (15.0h)** | 0.3103 | Other Exception | 26 |
| 2 | 894.37 min (14.9h) | 0.8088 | Other Exception | 27 |
| 3 | 891.35 min (14.9h) | 0.8134 | Other Exception | 26 |
| 4 | 887.73 min (14.8h) | 0.7593 | Other Exception | 26 |
| 5 | 887.57 min (14.8h) | 0.2885 | Other Exception | 41 |
| 6 | 887.50 min (14.8h) | 0.3167 | Other Exception | 42 |
| 7 | 887.38 min (14.8h) | 0.8134 | Other Exception | 26 |
| 8 | 886.18 min (14.8h) | 0.3688 | Other Exception | 29 |
| 9 | 886.17 min (14.8h) | 0.3688 | Other Exception | 29 |
| 10 | 885.90 min (14.8h) | 0.3028 | Other Exception | 30 |

> **Até 15 horas** de antecipação! Sessões com muitos logs (26-42) e loss moderado indicam anomalias sutis que o modelo captura muito antes da falha.

---

## 6. Top 20 — Maior Perda (Padrões Mais Anômalos)

| # | Alert Loss | Lead Time | Categoria |
|---|-----------|-----------|-----------|
| 1 | **10.6235** | 16.52 min | Other Exception |
| 2 | 10.6235 | 0.00 min | InterruptedIOException |
| 3 | 10.6235 | 0.02 min | InterruptedIOException |
| 4 | 10.6235 | 0.00 min | InterruptedIOException |
| 5 | 10.6235 | 116.33 min | Other Exception |
| 6 | 10.6217 | 1.23 min | Other Exception |
| 7 | 10.3776 | 120.62 min | Other Exception |
| 8 | 10.3776 | 0.00 min | InterruptedIOException |
| 9 | 10.3776 | 179.17 min | Other Exception |
| 10 | 10.3776 | 0.00 min | InterruptedIOException |

> Loss ~10.6 (350× o threshold) indica templates **nunca vistos** no treinamento — anomalias completamente fora da distribuição.

### Distribuição de Alert Loss (TP)

| Percentil | Alert Loss |
|-----------|-----------|
| P10 | 0.3119 |
| P25 | 0.3943 |
| **P50** | **0.4858** |
| P75 | 1.2057 |
| P90 | 1.2258 |
| P95 | 1.2258 |
| P99 | 5.8867 |

---

## 7. Análise de Falsos Negativos (Erros Perdidos)

| Categoria | Perdidos | Avg Session Size | Causa Provável |
|-----------|----------|-----------------|----------------|
| **InterruptedIOException** | 1,649 | 2 logs | Sessões curtas demais |
| **NameSystem/BlockMap** | 1,307 | 2 logs | Padrão idêntico ao normal |
| Other Exception | 23 | 28 logs | Anomalias sutis |
| EOFException | 3 | 53 logs | Raros, baixa representação |
| SocketTimeoutException | 1 | 44 logs | Caso isolado |

> [!CAUTION]
> **97.9% dos FN** são sessões com **apenas 2 logs**. O modelo precisa de contexto suficiente para distinguir anomalias — sessões ultra-curtas não geram perplexidade diferenciável.

---

## 8. Análise de Falsos Positivos

| Métrica | Valor |
|---------|-------|
| Total FP | 733 |
| FP Loss Range | 0.2885 – 0.6194 |
| FP Avg Loss | 0.3270 |
| FP Median Loss | 0.3103 |

> FPs têm loss próximo ao threshold (0.2863), indicando que são **borderline** — sessões normais com padrões levemente incomuns. Taxa de FP extremamente baixa (**1.3%** das sessões normais).

---

## 9. Resumo Executivo

### ✅ Pontos Fortes
- **Precision 95%**: Quando o modelo alerta, está certo 95% das vezes
- **Specificity 98.7%**: Quase nenhum falso alarme em sessões normais
- **Lead time até 15h**: Capacidade extraordinária de antecipação
- **Other Exception recall 99.8%**: Categoria dominante quase perfeita

### ⚠️ Pontos de Atenção
- **NameSystem/BlockMap (recall 0%)**: 1,307 sessões completamente indetectáveis — padrão visualmente idêntico ao normal
- **Sessões ultra-curtas (2 logs)**: 2,956 dos 2,983 FN (99.1%) têm ≤2 logs — limitação fundamental da abordagem sequencial
- **InterruptedIOException (recall 66.5%)**: Metade das sessões curtas não tem contexto suficiente

### 📌 Recomendações
1. **Regra complementar**: Para sessões com ≤2 logs, usar classificação por template (rule-based) ao invés de perplexidade
2. **Threshold refinado**: Testar threshold menor (~0.20) para capturar anomalias borderline, aceitar mais FPs
3. **Agrupamento de sessões**: Combinar sessões do mesmo bloco que ocorrem dentro de janela temporal para enriquecer contexto
