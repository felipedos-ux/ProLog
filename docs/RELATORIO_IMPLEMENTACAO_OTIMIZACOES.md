# RELATÓRIO DE IMPLEMENTAÇÃO: TÉCNICAS DO ESTADO DA ARTE (SOTA)
## Transposição de Módulos Parser-Free para o Motor GPT-2 BPE

---

## 1. Visão Geral
Este documento reflete a análise técnica detalhada do documento _"Análise Comparativa Papers Similares.docx"_ somada a buscas profundas na literatura recente (2021-2026). O objetivo é traçar o mapa exato de implementação de técnicas _Parser-Free_ ou de Modelos de Linguagem para otimizar o nosso motor `GPT-2 BPE` e romper a barreira técnica em direção aos `95%+` de precisão (F1 Score).

---

## 2. Técnicas Arquiteturais Descobertas e a sua Implementação

### 2.1 Contrastive Learning (Baseado no ContraLog - ICLR 2026)
O **ContraLog** inovou ao forçar os embeddings de logs contínuos a se distanciarem, superando a discretização simples.
*   **O Conceito:** Treinar o modelo aproximando ("Negative/Positive Pairs") as sentenças normais e distanciando as anômalas no vetor latente usando _InfoNCE Loss_.
*   **Plano de Implementação no ProLog:**
    1.  Interceptar o vetor invisível da última camada do GPT-2.
    2.  Criar pequenos _batches_ híbridos (com amostras normais rotuladas + anomalias induzidas).
    3.  Modificar o loop de treino base para adicionar: `Total_Loss = Autoregressive_Loss (Cross-Entropy) + λ * Contrastive_Loss`.

### 2.2 Semantic Distance & Adaptive Threshold (Baseado no ADALog - 2025 e LAnoBERT - 2021)
Modelos não-supervisionados como **ADALog** e **LAnoBERT** rejeitaram thresholds fixos (ex. K=5). O LAnoBERT provou que a perplexidade do Token Masked (MLM Loss) é um preditor poderoso por si só. O ADALog expandiu isso via _Distance-Based Detection_.
*   **O Conceito:** O Threshold de corte para detectar anomalias é ajustado em tempo de execução (`Média + K * Desvio Padrão` das perplexidades conhecidas do dataset normal).
*   **Plano de Implementação no ProLog:**
    1.  Correr inferência sem otimizador sobre os dados normais.
    2.  Registrar o valor global do Cross-Entropy Loss de tudo.
    3.  Calcular Média (µ) e Desvio Padrão (σ).
    4.  Substituir o Threshold fixo atual (`10.0` ou Top-K) pela fórmula dinâmica: `Threshold = µ + 2.5 * σ`.

### 2.3 Expressões Regulares de Limpeza Parcial (Regex Mínimo - Baseado no LogLLM - 2024)
O **LogLLM** atingiu 99.7% usando o LLaMA gigantesco. A sacada genial não foi passar a frase crua, nem usar o Drain absoluto, mas um filtro híbrido: **Regex Mínimo**.
*   **O Conceito:** Transformar IPs dinâmicos ou Timestamping de nanosegundos abstratos em tokens uniformes genéricos como `<IP>` para não explodir o vocabulário (reduzindo a sub-tokenização excessiva). A semântica original (nome dos processos) permanece inteira.
*   **Plano de Implementação no ProLog:**
    ```python
    import re
    def minimal_regex_filter(log_text):
        # 1. Mascarar IPs Dinâmicos
        log_text = re.sub(r'\d{1,3}(\.\d{1,3}){3}', '<IP>', log_text)
        # 2. Mascarar Timestamp se houver 
        log_text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '<TS>', log_text)
        # 3. Hexadecimais como IDs de Transações
        log_text = re.sub(r'0x[a-fA-F0-9]+', '<HEX>', log_text)
        return log_text
    ```

### 2.4 Treinamento em Estágios (Domain Adaptation - Baseado no LogLLM - 2024)
O LogLLM demonstrou um F1 estupendo dividindo o Treinamento.
*   **O Conceito:** Fazer a máquina ler arquivos de log genéricos na Fase 1, e só na Fase 2 focar no Dataset específico (HDFS/OpenStack).
*   **Plano de Implementação no ProLog:** Criar o script de pré-treino onde alimentamos o GPT-2 primeiro com dados triviais do OpenStack sem anotações, e num segundo momento refinamos o `learning_rate` numa curva agressiva com amostras validadas do HDFS.

### 2.5 Extensão do Contexto Temporal (Baseado no LogFiT - 2024)
O **LogFiT** substituiu modelos sub-dimensionados pelo Longformer (capaz de devorar 4096 tokens) porque janelas pequenas perdiam contexto histórico que causavam as anomalias tardias.
*   **O Conceito:** Expansão direta da _Sliding Window_.
*   **Plano de Implementação no ProLog:** 
    Aumentar o Hyperparametro de `block_size` de `128` para `1024` no GPT-2 atual. Caso falte VRAM para processar sequencias muito massivas, adotar o design do NeuralLog: Cortar com passo (*step_size*) criando sub-janelas sobrepostas. `(Window=20 logs reais, Step=5)`.

---

## 3. RoadMap Específico de Modificação para o nosso Código 

Para injetar as descobertas acima e materializar as arquiteturas de 2024-2026, propomos o seguinte plano em nosso código (Workflow D e E):

1. **`preprocessing.py`**: Interceptar os dados brutos e aplicar o **Regex Mínimo** (`<IP>`, `<HEX>`, `<TS>`). Isso manterá o modelo livre do parser engessado `Drain`, mas aliviará o GPT-2 da carga de memorizar letras hexadecimais aleatórias.
2. **`config.py`**: Atualizar a `WINDOW_SIZE` para uma abordagem deslizante com overlap (LogFiT) e incluir um valor global `ADAPTIVE_K` para modelar o limite do Loss de Perplexidade (LAnoBERT/ADALog).
3. **`train_stage.py` (Novo)**: Implementar uma sub-rotina para rodar as predições no dataset sadio, tirar Média (`np.mean()`) + Desvio Padrão (`np.std()`), salvar isto num pickle, e amarrar a distância no `detect.py`.
4. **`models.py / core loss`**: Num futuro exp, injetar um Loss Personalizado (Contrastive Loss) penalizando o GPT-2 se o cluster dos Tensors Ocultos das sentenças suspeitas se aglutinar ao tensor dos saudáveis.

---

## 4. Conclusão da Investigação
Estes papéis confirmam amplamente o que diagnosticamos de forma independente no "Relatório HDFS" anterior:
**A abordagens de parser-Drain estão obsoletas para os limites modernos de contextualização.** 

A literatura acadêmica que atingia `90%+` nos últimos 3 anos foi pivotando completamente de IDs Discretos para *Embeddings Semântico Densos* (`NeuralLog/ContraLog`) e a Análise via Top-K fixos foi substituída por *Perplexidade e Adaptação Unsupervised* (`LAnoBERT/ADALog`).
O GPT-2 atual da nossa aplicação, acoplado com um Regex Mínimo + Loss Dinâmico adaptativo, possui absolutamente todas as "peças teóricas" para reproduzir a performance do LogLLM (2024) e LogFiT (2025).
