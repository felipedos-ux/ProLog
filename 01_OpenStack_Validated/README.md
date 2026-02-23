# LogGPT-Small: Predição de Falhas com LLM Customizado

Modelo de linguagem customizado (30M parâmetros) para detecção e antecipação de falhas em logs.

## 🤖 Modelo LLM

**Arquitetura**: GPT-2 Customizado (from scratch)
- **Parâmetros**: ~30M
- **Layers**: 4 transformer blocks
- **Attention Heads**: 4
- **Embedding Dimension**: 256
- **Context Window**: 128 tokens
- **Tokenizer**: [DistilGPT2](https://huggingface.co/distilgpt2) (vocab_size: 50,257)

**Treinamento**:
- Dataset: Logs normais do OpenStack (80% dos dados)
- Objetivo: Causal Language Modeling (predição do próximo token)
- Framework: PyTorch + Transformers

## 📊 Resultados

- **Taxa de Antecipação**: 88.2% (149/169 falhas)
- **Lead Time Médio**: 17.70 minutos
- **Lead Time Máximo**: 27.88 minutos
- **Recall**: 100%
- **F1-Score**: 0.8848

## 📁 Arquivos

- `model.py`: Arquitetura GPT customizada (4 layers, 4 heads, 256 dim)
- `dataset.py`: Preparação de dados (tokenização, chunking)
- `train_custom.py`: Script de treinamento
- `detect_custom.py`: Script de detecção e cálculo de lead time
- `model_weights/`: Modelo treinado (checkpoint)

## 🚀 Como Usar

### Treinar Modelo
```bash
python train_custom.py
```

**Configuração**:
- Batch Size: 8
- Épocas: 10
- Learning Rate: 3e-4
- Tempo: ~10 minutos (GPU)

### Executar Detecção
```bash
python detect_custom.py
```

**Saída**:
- Métricas de classificação (F1, Precision, Recall)
- Lead times por sessão
- Top 10 melhores/piores antecipações
- Análise de diversidade de falhas

## 🔧 Requisitos

**Treinamento**:
- GPU: NVIDIA RTX 3080 Ti (12GB) ou superior
- RAM: 16GB

**Produção**:
- CPU: 4 cores @ 2.5GHz (GPU opcional)
- RAM: 4GB
- Latência: < 1s por sessão

## 📖 Documentação Completa

Ver `../reports/loggpt_relatorio_detalhado.md` para:
- Algoritmo de detecção passo-a-passo
- Explicação do cálculo de lead time
- Exemplos práticos com código
