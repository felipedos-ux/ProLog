# 📘 LogGPT: Benchmarking e Análise de Previsão de Falhas em Logs

> **Status do Projeto**: 🔄 Fase 3 (HDFS Benchmark) em andamento.
> **Objetivo**: Validar a capacidade de modelos GPT (Next-Token Prediction) para antecipar falhas em sistemas distribuídos.

---

## 📑 Índice
1.  [Visão Geral do Projeto](#-visão-geral-do-projeto)
2.  [Fase 1: Validação no OpenStack (Sucesso)](#-fase-1-validação-no-openstack)
3.  [Fase 2: O Desafio do BGL (Análise Crítica)](#-fase-2-o-desafio-do-bgl)
4.  [Fase 3: HDFS Benchmark (Estado da Arte)](#-fase-3-hdfs-benchmark)
5.  [Referências Bibliográficas](#-referências-bibliográficas)
6.  [Estrutura do Repositório](#-estrutura-do-repositório)

---

## � Visão Geral do Projeto

Este projeto investiga a aplicação de Large Language Models (LLMs), especificamente a arquitetura **GPT-2**, para a análise de logs de sistemas. A hipótese central é que logs de software estruturado possuem uma "gramática" previsível, permitindo que modelos gerativos antecipem erros (Lead Time) ao detectar desvios na sequência esperada de eventos.

---

## ✅ Fase 1: Validação no OpenStack

**Objetivo**: Replicar os resultados do paper original LogGPT para garantir que nossa implementação (LogGPT-Small) funciona.

### 🔬 Metodologia
*   **Dataset**: OpenStack (Logs de Cloud Management).
*   **Abordagem**: Sessão por `Trace ID` (Logs agrupados por requisição HTTP/VM).
*   **Modelo**: GPT-2 Small (4 layers, 256 embedding).

### 📈 Resultados
*   **F1-Score**: **96.6%**
*   **Conclusão**: O modelo aprendeu perfeitamente a sequência de provisionamento de VMs (`Create` -> `Allocate Network` -> `Success`). Quando a sequência quebra, a perplexidade sobe, detectando a anomalia.

---

## ⚠️ Fase 2: O Desafio do BGL (BlueGene/L)

**Objetivo**: Aplicar a mesma lógica de "Previsão de Sessão" em logs de Supercomputadores (Hardware).

### ❌ O Problema
Ao tentar transferir o aprendizado do OpenStack para o BGL, encontramos uma barreira intransponível para previsão pura:
1.  **Ausência de Sessão**: BGL não tem `Trace ID`. Logs de hardware são contínuos streams de milhares de componentes.
2.  **Sessão Artificial (NodeID)**: Tentamos agrupar por Nó, mas isso cria "sessões" de meses de duração, sem início/fim claros.
3.  **Interleaving**: Eventos de falha (Hardware) acontecem aleatoriamente, sem uma cadeia causal de software anterior clara para o modelo "prever".

### 🔎 A Descoberta (Pesquisa)
Investigando a literatura para entender o fracasso, encontramos o paper **LogADEmpirical (ICSE 2022)**, que critica exatamente o que tentamos fazer.
*   *Citação*: "Muitos benchmarks anteriores inflaram resultados no BGL usando janelas incorretas."
*   *Veredito*: BGL requer **Sliding Windows** (janelas deslizantes de tempo/contagem) e classificação, não previsão de sessão.

### 🛠️ O Pivô Técnico
Mudamos a estratégia no BGL para:
*   **Abordagem Híbrida**: Janelas de 20 eventos + Detecção de Anomalia por Top-K (Adaptive).
*   **Resultado**: Melhoramos a detecção, mas concluímos que BGL **não é adequado** para testar *Previsão Generativa* (Lead Time).

---

## 🔄 Fase 3: HDFS Benchmark (Hadoop Distributed File System)

**Objetivo**: Provar a capacidade de previsão (Lead Time) em um ambiente adequado (Software Distribuído).

### 💡 Por que HDFS?
Baseado em **DeepLog (CCS'17)** e no próprio **LogGPT**, o HDFS é o padrão-ouro para modelos sequenciais porque:
1.  **Processos de Software**: Segue máquinas de estado finitas (`Allocation` -> `Packet` -> `Ack`).
2.  **Sessão Nativa**: O `BlockId` isola cada transação de arquivo, permitindo previsão contextual limpa.
3.  **Repetibilidade**: Vocabulário pequeno (~46 templates), ideal para o GPT aprender a "rotina" normal.

### 🚀 Status Atual
*   **Preprocessing**: Reescrevemos o parser para ler logs brutos (1.5GB) usando Regex otimizado (Polars).
*   **Validação**: Pipeline testado com dataset dummy (FPR 0.18%).
*   **Bloqueio**: Aguardando download do Ground Truth (`anomaly_label.csv`) para calcular Recall real.

---

## 📚 Referências Bibliográficas

As decisões técnicas deste projeto foram embasadas nos seguintes papers:

1.  **LogGPT: Log Anomaly Detection via GPT** (Nokia, 2023)
    *   *Uso*: Base da arquitetura e hiperparâmetros (Block Size=64, Embed=256). Valida uso de HDFS/OpenStack.
2.  **LogADEmpirical** (ICSE 2022)
    *   *Uso*: Foi crucial para entendermos por que nossa abordagem "Session-based" falhou no BGL e pivotarmos para Sliding Window.
3.  **DeepLog** (CCS 2017)
    *   *Uso*: Fundamentou a escolha do HDFS como o dataset ideal para testar modelos sequenciais (LSTM/Transformer).
4.  **Loghub** (GitHub Repo)
    *   *Uso*: Fonte dos datasets e templates de parsing.

---

## 📂 Estrutura do Repositório

```
D:\ProLog\
├── 01_OpenStack_Validated\   # ✅ Código validado (Fase 1)
├── 03_HDFS_Benchmark\        # 🔄 Pipeline HDFS atual (Fase 3)
├── 05_loggpt_bgl\            # 🧪 Laboratório de Experimentos BGL (Fase 2)
│   ├── reports\              # Relatórios detalhados da falha/pivô BGL
│   └── ...scripts...         # Scripts híbridos e window-based
├── data\                     # Datasets Brutos
└── README.md                 # Este documento
```
