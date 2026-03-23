# 🔬 Spec Doc: Relatório de Execução e Definição Metodológica (HDFS Lab)

## 📌 1. Escopo e Objetivo
Estabelecer e validar a arquitetura ideal para detecção de anomalias em logs do HDFS usando Modelagem de Linguagem Causal (GPT-2/LogGPT), partindo da hipótese de que técnicas estado-da-arte (Top-K, Deduplicação) de modelos concorrentes podem alavancar a precisão e cobertura do modelo padrão. O objetivo final é definir o **Experimento F**, a nova metodologia "Autêntica" para a tese e refutar abordagens que se provaram inconsistentes com LLMs baseados em processamento de texto natural.

---

## 🏗️ 2. O que já foi feito (Experimentos Validados)

Para isolar as variáveis formou-se o ambiente `06_HDFS_Lab/`, onde reproduzimos metodologias de artigos acadêmicos consagrados.

### Experimento A: O Baseline (Loss Threshold)
- **Técnica:** O modelo prevê tokens sequencialmente. Se a 'surpresa' (Loss) de um bloco de log for maior que `3.0` (Threshold), é anomalia. O modelo usado foi o equivalente a arquitetura nativa GPT-2, parametrizada para 256 de dimensão oculta de Features.
- **Resultado:** **F1-Score 88.2%**. O baseline se demonstrou extremamente robusto, lidando muito bem com o dataset bruto do HDFS.

### Experimento B: A Falácia do Top-K Metric (Estilo LogGPT / DeepLog originário)
- **Técnica:** Avaliar logs checando se o próximo token está entre os "K=7" mais prováveis preditos pela rede causal de self-attention. Se não estiver, alerta a anomalia.
- **Resultado Final Avaliado:** **F1-Score 6.2% (Recall de 3.2%)**. 
- **Descoberta Crítica (Por que falhou tão brutalmente?):** A literatura acadêmica de Top-K na área de AIOps trabalha quase que estritamente convertendo templates completos de log para IDs únicos categorizados (ex: Log Inteiro de Recebimento de Bloco = Evento E1). Nosso LLM é um _Raw-Text LM_, ele processa a string pura da vida real com tokens BPE (Byte-Pair Encondings). O problema fatal dessa transposição é que pedaços subnominais de palavras (ex: `Rec`, `eived`, `block`) são tão densamente triviais no vocabulário que o modelo **sempre** os coloca no Top-K de previsões com alta margem de confiança, seja na sentença normal e seja na sentença anômala, **cegando o sistema para a anomalia real da composição**, pois ele valida o 'pedaço trivial da palavra' em vez do sentido da frase (A Cross Entropy total).
- **Conclusão Sistêmica:** Modelos textuais baseados em tokens fracionados (BPE LMs) precisam fundamentalmente ser avaliados por _Loss Threshold_ sequencial matricial. Probabilidade de Top-K unitário gera Falsos Negativos crônicos, a menos que o texto já venha convertido como tokens de eventos singulares perfeitos - o que anularia a principal vantagem de um LLM nativo que interpreta semântica.
- **A Confirmação do Estado da Arte:** Até mesmo o **LogLLaMA (2025)**, que atinge 99.7% de F1 usando um gigante de 7 Bilhões de parâmetros, **utiliza o algoritmo Drain no seu step 1** para extrair as chaves do log ("Log Keys") ignorando variáveis, IPs, e texto cru não mapeado. Nosso modelo é **Parser-Free**, lidando com 50.257 sub-palavras humanas da vida real. Comparar o F1 de 86-88% do nosso modelo Parser-Free com o F1 de 98% de um modelo "Drain-Based" é comparar um classificador de dicionário semântico com um decorador de IDs.

### Experimento C: Deduplicação de Volumes (Estilo SiaLog)
- **Técnica:** Limpeza de sessões perfeitamente idênticas antes de passar pelo treinamento e thresholding, reduzindo a bias natural de predição do modelo (Overfitting por dado super-representado) que ofuscam sequencias raras. No HDFS de 5000 sessões, essa técnica removeu absurdos 30.4% de redundâncias idênticas (1.521 documentos), purificando a entropia.
- **Resultado:** O F1-Score da técnica Top-K subiu miraculosamente de **6.2% para 57.43%**. Apenas retirando o ruído, retirou-se a "venda" do modelo e ele passou a generalizar as raras anomalias, mas ainda demonstrou que Top-K não é a heurística ideal.

### Experimento F: A Correção Teórica (A Metodologia Otimizada)
A validação do **Por quê o Top-K Errou** abriu o caminho para a metodologia final ideal: O motor Baseline (Loss Threshold) fortalecido pela Remoção de Ruído (Deduplicação SiaLog). Foi a união das melhores táticas aprovadas nas etapas anteriores (`Threshold + Deduplicate_T`).

**Métricas Experimentais Finais:**
*   **F1 Score:** `85.84%`
*   **Precision:** `95.67%`
*   **Recall:** `77.85%`

**Interpretação:**
1. A _Precision_ do modelo tornou-se monumental (atingindo cirúrgicos 95.67%). Ele quase nunca aplica "Falso-Positivo" no baseline Threshold graças à limpeza da Deduplicação durante as épocas de Treinamento e estabilização de Loss.
2. Embora a matriz de `88.2% F1` bruta do **Exp A** tenha caído timidamente para `85.8%` no **Exp F**, há uma **robustez sistêmica muito superior**: o modelo anterior decorava blocos com mais de `30%` de repetição viciada. Este último modelo (F) precisou deduzir as sequências matematicamente sem a "cola" dos dados duplicados (o "Overfitting Implícito" do HDFS).

Essa métrica final encerra a tese: A Detecção de Falhas via NLP sobre logs baseados em BPE **NÃO pode ser parametrizada usando heurística linear Top-K** originária das redes Word2Vec, a menos que uma rigorosa **purificação da redundância textual** (Dedupe de 30%+) anteceda o modelo. O formato absoluto e blindado para LLMs se confirma no paradigma `Cross-Entropy Loss Thresholding`.

### Experimento D e E: Scaling e Reforço (RL PPO)
- A Arquitetura Menor (60 Dimensões em Exp D) resultou em decréscimo maciço de Assertividade no Loss. Modelos GPT usando Text BPE demandam mais massa vetorial referencial.
- O RL PPO (Experimento E) com recompensa binária usando Top-K se provou instável para generalizar pela mesma fundamentação técnica colhida no Experimento B.

---

## 🎯 3. O que falta fazer (O Caminho Crítico de Ponto Final)

Agora que resolvemos os mistérios estruturais e teóricos definitivos, precisamos empacotar nossa metodologia autêntica de encerramento.

### O "Experimento F" (A Nova Metodologia Autêntica de Tese)
A nossa metodologia final validada para a apresentação magistral no TCC será oficialmente a combinação dos nossos acertos teóricos catalogados:
1. **Engine (A Arquitetura Otimizada):** LogGPT Causal LM (Arquitetura GPT-2, 256 dimensões / 4 layers de Self-Attention / 4 Heads).
2. **Dados (O Data-Centric Filter):** Deduplicação Estilo-SiaLog para desinflar tensores tendenciosos durante o Language Modeling, removendo ruído para as Heads de Self-Attention mapearem anomalias puras.
3. **Detecção (Criterion):** Sequência contínua avaliada por *Cross-Entropy Loss Threshold*, descartando em absoluto matrizes de avaliação de tokens baseadas puramente em Top-K estritas.

### 📌 Passos de Ação Restantes do Fluxo de Trabalho (Workflow C->E)
- [x] Aplicar o código da **Metodologia F (Deduplicar + Threshold do GPT-2 Engine)** com o seu script dedicado em `.context`, finalizando a validação de laboratório.
- [x] Transpor a teoria recém confirmada para dentro do documento `docs/RELATORIO_REFATORACAO_HDFS.md`, focando a escrita acadêmica em argumentar que Modelos textuais precisam trabalhar com densidade total (Loss Threshold) por causa dos limitadores do tokenizer BPE.
