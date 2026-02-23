# Guia Definitivo e Detalhado de Replicação LogGPT: HDFS & OpenStack

**Aviso para a IA Assistente:** Você foi instruída a replicar uma arquitetura de detecção de anomalias em logs baseada em **Causal Language Modeling (LogGPT/GPT-2)**. O dataset alvo (como o OpenStack) possui um problema grave documentado como **Zero-Overlap**: os templates dos logs anômalos não possuem NENHUMA intersecção com os templates do treino normal. 

Se você usar métodos baseados em "Cross-Entropy Threshold" (onde a anomalia é detectada por ter uma 'loss' média/máxima maior que um limite), seu modelo **FALHARÁ** (F1-score ≈ 0 e 100% de Falsos Positivos). A IA "se surpreende" (alta cross-entropy) tanto com eventos inéditos de erro quanto com eventos normais inéditos em nova ordem.

Para solucionar isso, você **DEVE** seguir a abordagem abaixo (Top-K Parameter-Free baseada em Sequência de Eventos). Siga os passos minuciosamente.

---

## Passo 1: O Formato dos Dados (EventId vs EventTemplate)

**O Problema Clássico:** 
Construir o input do GPT passando strings longas (ex: `"Failed password for root from 192.168.1.1 port 22"`) faz o modelo focar excessivamente na semântica de "subwords" e perder a percepção cronológica do log como um evento discreto no tempo.

**A Solução Obrigatória:**
Você deve pré-processar o dataset para extrair apenas a ID estática do template de log (a `EventId`, que costuma ser uma string curta, ex: `E1`, `E45`, `E200`).
Você deve agrupar essas IDs pelo identificador de sessão cronológica (`SessionId` ou `test_id`) em uma única string, separada por espaços.

**Ação da IA (Exemplo em Polars/Pandas):**
1. Filtre a coluna do identificador (assegure que não é Nula).
2. Agrupe pela Sessão (`test_id`).
3. Ao construir a label da sessão anômala através da coluna `anom_label`, você **DEVE USAR `.max()`**. Atenção a este bug: usar `.first()` falhará porque sessões de erro muitas vezes iniciam com logs normais antes do travamento (label `0` seguida de label `1`).
4. Junte as sequências com `.join(" ")`.

```python
# Como deve ficar o DataFrame de Sessões:
# test_id | label | EventTemplate (String de EventIds)
# 100     | 0     | "E1 E2 E2 E5 E1"
# 101     | 1     | "E1 E2 E90 E91 E22"  <-- E90 e E91 nunca apareceram no treino
```

---

## Passo 2: Tokenização Causal e `Dataset` PyTorch

**O Problema Clássico:**
Muitas replicações usam a função `group_texts` do HuggingFace (que concatena TODOS os logs do dataset em um linguição e fatia arbitrariamente em blocos de `128` ou `256` tokens). Isso destrói a coesão da "sessão" e força o modelo a memorizar os blocos em vez de entender o que é o início, meio e fim de um atendimento lógico do servidor.

**A Solução Obrigatória:**
Cada "Sessão" deve ser **1 (um) único e independente exemplo de treinamento**.

**Ação da IA (Código do PyTorch Dataset):**
Você deverá configurar o Tokenizer do GPT-2 para preencher (PAD) com o EOS token.
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # ID 50256
```

Sua classe `Dataset` precisa fatiar no `MAX_LENGTH` máximo do GPT-2 (1024), pois as sessões podem ser longas (no OpenStack, chegam a ~494 eventos).

```python
class LogSessionDataset(Dataset):
    def __init__(self, sessions_df, tokenizer):
        self.data = sessions_df
        self.tokenizer = tokenizer
        self.max_len = 1024 # Limite do modelo base

    def __getitem__(self, idx):
        row = self.data.iloc[idx] # Para Pandas, .row para Polars
        seq_string = row['EventTemplate'] # "E1 E2 E5 E1"
        
        # Tokeniza os IDs inteiros da sessão
        tokens = self.tokenizer.encode(seq_string, truncation=True, max_length=self.max_len)
        return torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch):
    # DYNAMIC PADDING: Só preenche o lote até a maior sessão presente NELE.
    max_batch_len = max(len(x) for x in batch)
    # Preenche o tensor 2D com 50256 (pad_token)
    padded = torch.full((len(batch), max_batch_len), 50256, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded
```

---

## Passo 3: O Causal LM Shift (Treinamento)

**Ação da IA (Loop de Treino):**
Durante o `train()`, como você não usará a classe de trainer automática do HuggingFace, você passará todo o tensor de tokens pela rede gerando os Logits, e usará Cross-Entropy para calcular o loss manual. O segredo do Next-Word-Predictor Causal:

1. O modelo é treinado usando o otimizador `AdamW` bruto sem modificações.
2. Não é necessário Regularizadores agressivos (como L2 Decay altíssimo ou Label Smoothing) porque, no nível de EventId, o GPT-2 não sofre de sobreajuste crítico.
3. Você deve desalinhar (Shift) o tensor manual em `1` posição.

```python
for batch in train_loader:
    # batch shape é [Batch_Size, T] (onde T é o tempo/comprimento sequencial)
    
    # O contexto lido (passado) é tudo menos o último elemento
    inp = batch[:, :-1].to(device)  # shape: [B, T-1]
    
    # O evento que a rede deve prever é tudo a partir do segundo elemento
    tgt = batch[:, 1:].to(device)   # shape: [B, T-1]
    
    logits, loss = model(inp, targets=tgt) # model encapsula F.cross_entropy(logits, tgt)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Passo 4: O Coração do Sistema — Detecção Top-K (MUITO IMPORTANTE)

**O Problema Clássico (Threshold e Mean Loss):**
A IA costuma criar um script `calibrate.py` que calcula a média da entropia de toda a validação e define um limiar global (por exemplo, "tudo com perda > 3.0 é anomalia"). Isso **Gera 100% de FP** num cenário de *zero-overlap*, onde eventos normais levemente embaralhados explodem a entropia assim como os anômalos inéditos.

**A Solução Obrigatória:**
Você **DEVE** deletar qualquer lógica de limite/threshold estatístico linear. Você usará a avaliação **Parameter-Free Top-5**. 

*Como funciona?*
Na inferência, para TODOS os passos temporais `t` de dentro da sessão, o LogGPT calculará o Softmax das predições de todo o vocabulário e entregará quais seriam os **5 eventos com a maior chance estatística** de vir logo em seguida do que aconteceu em `t`.
Se o Evento subsequente real que de fato ocorrer no arquivo `t+1` **NÃO ESTIVER CONTIDO** nas 5 opções estimadas do modelo, então `t` é categorizado como uma quebra fundamental de lógica estrutural do sistema.
Se ocorrer **1 ou mais quebras (anomalias em t)** durante 1 sessão inteira, a sessão INTEIRA vira 1 (anômala).

**Ação da IA (O Código de Inferência - Execute Exatamente Assim):**

```python
K = 5 # Defina Top-K

for batch in test_loader:
    input_ids = batch['input_ids'].to(device) # [B, T]
    
    with torch.no_grad():
        logits, _ = model(input_ids) # Sem shift aqui: Output logits shape [B, T, Vocab_Size]
        
        # Faz os mesmos Shifts para Pareamento de Resposta
        targets = input_ids[:, 1:]    # O evento que aconteceu de verdade: [B, T-1]
        preds = logits[:, :-1, :]     # A predição do modelo baseada no evento anterior: [B, T-1, Vocab_Size]
        
        # Tira o Softmax de probabilidade e pega os K Maiores Índices por posição
        probs = torch.softmax(preds, dim=-1)
        _, topk_inds = torch.topk(probs, K, dim=-1) # Shape resultante: [B, T-1, K]
        
        # VERIFICAÇÃO PRINCIPAL: Confere booleanamente se o Evento Real (`targets`) aparece em alguma das K colunas
        # Precisamos de .unsqueeze(-1) para virar [B, T-1, 1] e igualar de matriz contra matriz
        matches = (topk_inds == targets.unsqueeze(-1)).any(dim=-1) # Shape: [B, T-1] Booleano
        
        # TRATAMENTO DE PADDING: Ignore qualquer avaliação sobre os PAD tokens (50256) no final 
        target_mask = (targets != 50256)
        
        # DEFINE ANOMALIA: Um evento é anômalo se a predição NÃO ESTIVER NO MATCH (~matches) E não for um padding final.
        valid_anomalies = (~matches) & target_mask # Shape: [B, T-1] Booleano
        
        # DEFINE A SESSÃO: Um Batch[b] inteiro é 1 se HOUVER QUALQUER (.any()) "True" na dimensão do tempo (1)
        is_anom_pred = valid_anomalies.any(dim=1).cpu().numpy() # [B] -> Array booleano que diz se a sessão [0, 1, 2...] é Anomalia!
        
        # (Opcional) Capturando a Lead-Time ou o Primeiro Evento em Que ocorreu a falha detectada
        primeiro_passo_anomalo = valid_anomalies.int().argmax(dim=1).cpu().numpy()
        
        # is_anom_pred será sua coluna "y_pred_label"
```

Se a IA implementar fielmente o **Passo 4**, a detecção no OpenStack baterá os impressionantes ~92.35% de F1-Score com taxa minúscula de Falsos Positivos frente a cenários de limiar cego. Boa sorte na replicação!
