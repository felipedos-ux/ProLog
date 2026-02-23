# 🔬 Mega Relatório: Log Anomaly Detection com LogGPT

**Data**: 2026-02-07  
**Projeto**: ProLog - Universal Log Anomaly Detection  
**Status**: Em desenvolvimento, enfrentando problemas de generalização

---

## 📋 Índice

1. [Contexto e Objetivo](#1-contexto-e-objetivo)
2. [O Que Foi Feito](#2-o-que-foi-feito)
3. [Resultados no OpenStack (Sucesso)](#3-resultados-no-openstack-sucesso)
4. [Resultados no BGL (Falha)](#4-resultados-no-bgl-falha)
5. [Análise Comparativa](#5-análise-comparativa)
6. [Problemas Identificados](#6-problemas-identificados)
7. [Tentativas de Solução](#7-tentativas-de-solução)
8. [Perguntas para Pesquisa Externa](#8-perguntas-para-pesquisa-externa)

---

## 1. Contexto e Objetivo

### 1.1 O Problema

Sistemas computacionais geram milhões de logs diariamente. Detectar anomalias nesses logs é crucial para:
- Identificar falhas antes que causem interrupções
- Detectar intrusões de segurança
- Monitorar saúde do sistema

### 1.2 Abordagem Escolhida

Implementar um **detector universal de anomalias** baseado em **LogGPT** (Language Model) que:
1. Aprende padrões normais de execução
2. Detecta desvios como anomalias
3. Funciona em diferentes datasets sem retreino

### 1.3 Arquitetura LogGPT-Small

```
Modelo: GPT-style Transformer (Decoder-only)
- Layers: 4
- Heads: 4  
- Embedding: 256
- Block Size: 128
- Vocab: 50,357 (GPT-2 tokenizer)
- Parâmetros: ~3M
```

### 1.4 Datasets Utilizados

| Dataset | Origem | Logs | Anomalias | Tipo |
|---------|--------|------|-----------|------|
| **OpenStack** | Cloud Computing | ~200K | ~5% | IaaS operations |
| **BGL** | Blue Gene/L Supercomputer | ~4.7M | ~7.3% | HPC messages |

---

## 2. O Que Foi Feito

### 2.1 Fase 1: Treinamento no OpenStack

**Objetivo**: Treinar LogGPT-Small em sequências normais do OpenStack.

**Processo**:
1. Pré-processamento com Drain para extração de templates
2. Agrupamento por sessão (BlockId)
3. Treino com language modeling (next-token prediction)
4. 100 epochs, LR=1e-4, Batch=16

**Resultado**: ✅ **Sucesso** - F1 > 0.95 no OpenStack

### 2.2 Fase 2: Avaliação no BGL (Sem Retreino)

**Objetivo**: Validar se o modelo treinado no OpenStack generaliza para BGL.

**Processo**:
1. Pré-processamento do BGL com sliding window (20 eventos)
2. Avaliação usando perplexidade como sinal de anomalia
3. Threshold automático no validation set

**Resultado**: ❌ **Falha** - F1 = 0.65, TN = 0 (todos normais marcados como anomalias)

### 2.3 Fase 3: Fine-tuning no BGL

**Objetivo**: Fine-tune do modelo no BGL para ver se melhora.

**Processo**:
1. Fine-tune com 5,000 sequências normais do BGL
2. 100 epochs, LR=1e-6 (parâmetros do paper LogGPT)
3. Convergência excelente (loss 3.65 → 0.025)

**Resultado**: ❌ **Falha** - F1 = 0.65 (idêntico ao modelo universal)

### 2.4 Fase 4: Universal Detector Multi-Signal

**Objetivo**: Combinar múltiplos sinais além de perplexidade.

**Sinais Implementados**:
- **Perplexity**: Quão "surpreendente" é a sequência para o modelo
- **Rarity**: Frequência dos templates (templates raros = mais anômalos)
- **Context**: Consistência contextual da sequência

**Resultado**: ❌ **Falha** - F1 = 0.65, calibração automática falhou

---

## 3. Resultados no OpenStack (Sucesso)

### 3.1 Métricas Alcançadas

| Métrica | Valor | Target |
|---------|-------|--------|
| **Precision** | 0.96 | > 0.90 ✅ |
| **Recall** | 0.94 | > 0.90 ✅ |
| **F1 Score** | 0.95 | > 0.90 ✅ |
| **Accuracy** | 0.97 | > 0.95 ✅ |

### 3.2 Por Que Funcionou

1. **Dataset Homogêneo**: OpenStack tem padrões claros e repetitivos
2. **Vocabulário Limitado**: ~500 templates únicos
3. **Sessões Bem Definidas**: BlockId agrupa operações relacionadas
4. **Separação Clara**: Perplexidade de anomalias >> perplexidade de normais

### 3.3 Distribuição de Perplexidade no OpenStack

```
Normal:    [===1.0---2.0---3.0===]  μ=2.0, σ=0.5
Anomaly:            [===6.0---8.0---10.0===]  μ=8.0, σ=1.5
                                              
Separação: Δμ = 6.0 (EXCELENTE)
```

---

## 4. Resultados no BGL (Falha)

### 4.1 Métricas Obtidas

| Métrica | Universal | Fine-tuned | Target |
|---------|-----------|------------|--------|
| **Precision** | 0.489 | 0.489 | > 0.70 ❌ |
| **Recall** | 1.000 | 1.000 | > 0.80 ✅ |
| **F1 Score** | 0.657 | 0.657 | > 0.70 ❌ |
| **TN** | 0 | 0 | > 0 ❌ |
| **FP** | 511 | 511 | < 100 ❌ |

### 4.2 O Problema Central

**Threshold = 0.00** → Todos os casos são marcados como anomalias!

Isso significa que o algoritmo de busca de threshold encontrou que a **melhor** estratégia é marcar **TUDO** como anomalia, porque:
- Recall = 100% (todas anomalias detectadas)
- Precision = 48.9% (proporção de anomalias no dataset)
- F1 = 65.7% (melhor que qualquer threshold positivo)

### 4.3 Distribuição de Perplexidade no BGL

**Modelo Universal (OpenStack)**:
```
Normal:    [===2.7---3.4---4.1===]  μ=3.41, σ=0.36
Anomaly:         [===3.3---4.5---5.7===]  μ=4.51, σ=0.62
                      ^^^ OVERLAP ^^^
Separação: Δμ = 1.10 (INSUFICIENTE)
```

**Modelo Fine-tuned (BGL)**:
```
Normal:    [===0.4---1.1---1.8===]  μ=1.10, σ=0.36
Anomaly:         [===0.8---3.0---5.1===]  μ=2.98, σ=1.08
                      ^^^ OVERLAP ^^^
Separação: Δμ = 1.88 (AINDA INSUFICIENTE)
```

### 4.4 Por Que Não Funcionou

1. **Vocabulário Muito Maior**: BGL tem 242 templates únicos vs ~500 do OpenStack
2. **Templates Diferentes**: Nenhum overlap entre templates BGL e OpenStack
3. **OOV (Out-of-Vocabulary)**: Todos templates BGL são "desconhecidos" para o modelo
4. **Sobreposição de Distribuições**: Normal e anomalia têm PPL similar
5. **Dataset Desbalanceado**: 48.9% anomalias vs 51.1% normais

---

## 5. Análise Comparativa

### 5.1 Diferenças Estruturais entre Datasets

| Característica | OpenStack | BGL |
|----------------|-----------|-----|
| **Origem** | Cloud IaaS | Supercomputer |
| **Templates Únicos** | ~500 | 242 |
| **Overlap de Vocabulário** | - | 0% |
| **Formato de Sessão** | BlockId | Sliding Window |
| **Taxa de Anomalia** | ~5% | 48.9% (nas amostras) |
| **Tipo de Log** | Operações CRUD | Mensagens de sistema |
| **Estrutura** | `[timestamp] [level] message` | `[timestamp] [node] [type] message` |

### 5.2 Exemplo de Logs

**OpenStack**:
```
2024-01-15 10:23:45 INFO nova.compute.manager [req-abc] Starting instance i-12345
2024-01-15 10:23:46 INFO nova.compute.manager [req-abc] Instance i-12345 spawned successfully
```

**BGL**:
```
1117838570 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO generating core.12345
1117838570 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL double hummer exception
```

### 5.3 Por Que a Transferência Não Funciona

```mermaid
flowchart LR
    subgraph OpenStack
        A[Template: "Starting instance <*>"] --> B[Tokenizado: ID 1234]
        C[Template: "Instance <*> spawned"] --> D[Tokenizado: ID 5678]
    end
    
    subgraph BGL
        E[Template: "generating core.<*>"] --> F[Tokenizado: IDs desconhecidos]
        G[Template: "double hummer exception"] --> H[Tokenizado: IDs desconhecidos]
    end
    
    B & D --> I[Modelo aprende padrões]
    F & H --> J[Modelo não reconhece]
    
    I --> K[✅ Boa predição]
    J --> L[❌ Alta perplexidade para TUDO]
```

---

## 6. Problemas Identificados

### 6.1 Problema 1: Zero Transfer Learning

**Descrição**: O modelo treinado no OpenStack não transfere conhecimento para BGL.

**Evidência**:
- Perplexidade alta para TODOS os logs do BGL (normais e anômalos)
- Nenhum template do BGL aparece no vocabulário aprendido

**Causa Provável**:
- Tokenização baseada em texto (GPT-2 tokenizer) não captura semântica de logs
- Domínios completamente diferentes (cloud vs HPC)

### 6.2 Problema 2: Perplexidade Não Discrimina

**Descrição**: Mesmo após fine-tuning, perplexidade não separa normal de anomalia.

**Evidência**:
- Universal: Normal PPL 3.41, Anomaly PPL 4.51 (Δ=1.10)
- Fine-tuned: Normal PPL 1.10, Anomaly PPL 2.98 (Δ=1.88)
- Overlap significativo em ambos

**Causa Provável**:
- Anomalias no BGL não são "linguisticamente diferentes" dos normais
- Anomalias são definidas por padrões SEQUENCIAIS, não por tokens individuais

### 6.3 Problema 3: Calibração Automática Falha

**Descrição**: O sistema de calibração automática de threshold converge para 0.

**Evidência**:
- Melhor F1 ocorre com threshold=0 (marcar tudo como anomalia)
- Nenhum threshold positivo melhora o F1

**Causa Provável**:
- Dataset altamente desbalanceado (48.9% anomalias)
- Sobreposição de distribuições impede separação

### 6.4 Problema 4: Sliding Window Pode Ser Inadequada

**Descrição**: Janelas de 20 eventos podem não capturar contexto suficiente.

**Evidência**:
- Papers usam janelas maiores (60 eventos) ou baseadas em tempo (1 hora)
- Janelas curtas fragmentam padrões de erro

**Causa Provável**:
- BGL tem logs de alta frequência (milhões de mensagens)
- Erros se propagam por várias mensagens consecutivas

### 6.5 Problema 5: Sinal Único Insuficiente

**Descrição**: Usar apenas perplexidade como sinal não é suficiente para BGL.

**Evidência**:
- Multi-signal (perplexity + rarity + context) também falhou
- Calibração automática de pesos convergiu para {context: 0.8, outros: 0.1}

**Causa Provável**:
- BGL requer análise de padrões temporais, não apenas linguísticos
- Anomalias são detectáveis por sequência de eventos, não eventos individuais

---

## 7. Tentativas de Solução

### 7.1 Tentativa 1: Multi-Signal Fusion ❌

**O que fizemos**:
- Implementar 3 sinais: perplexity, rarity, context
- Calibração automática de pesos no validation set
- Fusão ponderada dos sinais

**Resultado**:
- Pesos calibrados: {perplexity: 0.1, rarity: 0.1, context: 0.8}
- F1 = 0.657 (sem melhoria)
- TN = 0 (todos normais marcados como anomalias)

**Por que falhou**:
- Sinal de context também não discrimina bem
- Calibração automática convergiu para solução degenerada

### 7.2 Tentativa 2: Fine-tuning no BGL ❌

**O que fizemos**:
- Fine-tune com 5,000 sequências normais
- 100 epochs, LR=1e-6 (parâmetros do paper)
- Convergência excelente (loss 3.65 → 0.025)

**Resultado**:
- Perplexidade reduzida (Normal: 3.41→1.10, Anomaly: 4.51→2.98)
- F1 = 0.657 (idêntico ao universal)
- Threshold ainda = 0

**Por que falhou**:
- Fine-tuning reduziu PPL absoluta mas não melhorou separabilidade
- Modelo aprendeu BGL, mas anomalias ainda têm PPL similar a normais

### 7.3 Tentativa 3: Ajuste de Window Size ⏳

**O que planejamos**:
- Testar janelas de 60 eventos (como no paper LogADEmpirical)
- Testar janelas baseadas em tempo (1 hora)

**Status**: Não executado ainda

### 7.4 Tentativa 4: Pesos Manuais ⏳

**O que planejamos**:
- Testar pesos manuais: {perplexity: 0.5, rarity: 0.3, context: 0.2}
- Normalizar sinais antes da fusão

**Status**: Não executado ainda

---

## 8. Perguntas para Pesquisa Externa

### 🔴 Perguntas Críticas (Prioridade Alta)

#### P1: Como papers de referência avaliam LogGPT no BGL?

**Contexto**: O paper do LogGPT reporta resultados no BGL, mas não conseguimos reproduzir.

**O que preciso saber**:
1. Qual pré-processamento exato usam para BGL?
2. Qual tamanho de janela (window size) usam?
3. Como definem o threshold de anomalia?
4. Usam apenas perplexidade ou múltiplos sinais?
5. Qual F1 reportam no BGL?

#### P2: Qual a metodologia de avaliação padrão para BGL?

**Contexto**: Diferentes papers usam metodologias diferentes.

**O que preciso saber**:
1. Sliding window vs session-based: qual é o padrão?
2. Tamanho de janela recomendado (10, 20, 60 eventos)?
3. Como lidam com janelas sobrepostas?
4. Usam validação temporal (train antes de test)?

#### P3: Como outros detectores LLM-based tratam templates OOV?

**Contexto**: Templates do BGL são completamente diferentes do OpenStack.

**O que preciso saber**:
1. Usam tokenização baseada em template ou em caractere?
2. Como lidam com templates nunca vistos no treino?
3. Aplicam smoothing para templates raros?
4. Usam embeddings semânticos ao invés de tokenização?

### 🟡 Perguntas Importantes (Prioridade Média)

#### P4: Qual é o state-of-the-art para BGL atualmente?

**Contexto**: Precisamos de um baseline para comparar.

**O que preciso saber**:
1. Top 3 métodos com melhor F1 no BGL (2023-2024)
2. Quais sinais/features usam?
3. Usam deep learning ou métodos clássicos?
4. Código disponível para reprodução?

#### P5: Como funciona a detecção de padrões sequenciais em logs?

**Contexto**: Anomalias no BGL parecem ser sequenciais, não pontuais.

**O que preciso saber**:
1. Métodos que detectam sequências anômalas (não eventos individuais)
2. DeepLog, LogAnomaly, LogRobust: como lidam com sequências?
3. Attention weights podem indicar anomalias?
4. Autoencoders sequenciais são melhores que LLMs?

#### P6: Fine-tuning vs Treino from Scratch: qual é melhor para logs?

**Contexto**: Fine-tuning não melhorou nossos resultados.

**O que preciso saber**:
1. Papers comparam fine-tuning vs treino do zero?
2. Qual learning rate ideal para fine-tuning em logs?
3. Quantas amostras são necessárias para fine-tuning efetivo?
4. Transfer learning de logs funciona entre domínios diferentes?

### 🟢 Perguntas Exploratórias (Prioridade Baixa)

#### P7: Embedding-based detection é melhor que perplexity-based?

**Contexto**: Perplexidade não está funcionando.

**O que preciso saber**:
1. Métodos que usam embeddings ao invés de perplexidade
2. Clustering de embeddings para detectar anomalias
3. Distância de embeddings como sinal de anomalia
4. BERT vs GPT: qual gera melhores embeddings para logs?

#### P8: Existe um "universal log detector" que funciona em múltiplos datasets?

**Contexto**: Nosso objetivo é um detector universal.

**O que preciso saber**:
1. Papers que avaliam em múltiplos datasets (BGL, HDFS, OpenStack, Thunderbird)
2. Métodos que não requerem retreino por dataset
3. Zero-shot ou few-shot detection em logs
4. Técnicas de domain adaptation para logs

#### P9: Como o LogADEmpirical avalia métodos no BGL?

**Contexto**: Paper de benchmark importante.

**O que preciso saber**:
1. Configuração exata para BGL (window size, step size)
2. Resultados de DeepLog, LogAnomaly, LogRobust no BGL
3. Pré-processamento (Drain parameters, grouping strategy)
4. Splits de treino/val/test

#### P10: Técnicas de data augmentation para logs anômalos

**Contexto**: Dataset desbalanceado (48.9% anomalias nas amostras).

**O que preciso saber**:
1. SMOTE para logs funciona?
2. Como gerar logs anômalos sintéticos?
3. Oversampling vs undersampling: qual é melhor?
4. Contrastive learning para logs: papers relevantes?

---

## 📊 Resumo do Status Atual

| Componente | Status | Problema |
|------------|--------|----------|
| LogGPT-Small | ✅ Implementado | - |
| Treino OpenStack | ✅ F1=0.95 | - |
| Avaliação BGL | ❌ F1=0.65 | TN=0, threshold=0 |
| Fine-tuning BGL | ❌ F1=0.65 | Não melhorou |
| Multi-Signal | ❌ F1=0.65 | Calibração falhou |
| Universal Detector | ❌ Não funciona | Não generaliza |

---

## 🎯 O Que Esperamos da Pesquisa

Com as respostas às perguntas acima, esperamos:

1. **Identificar o gap metodológico**: O que estamos fazendo diferente dos papers?
2. **Validar nossa abordagem**: Perplexidade é realmente a métrica certa?
3. **Encontrar alternativas**: Quais outros sinais/métodos funcionam no BGL?
4. **Definir próximos passos**: Ajustar janela? Mudar para embeddings? Usar outro modelo?

---

## 📚 Referências que Já Consultamos

1. **LogGPT** (arXiv 2302.07714): Parâmetros de treino
2. **DeepLog**: Arquitetura LSTM, window size 10
3. **LogAnomaly**: Template2Vec, LSTM
4. **LogADEmpirical**: Benchmark de múltiplos métodos

---

*Documento gerado para auxiliar pesquisa externa. Aguardando respostas para prosseguir com desenvolvimento.*
