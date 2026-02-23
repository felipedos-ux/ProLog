# Relatório Técnico: Análise e Guia de Implementação OpenStack

**Data:** 05 de Fevereiro de 2026
**Arquivo Analisado:** `OpenStack_data_original.csv`
**Total de Logs Processados:** 217.534

---

## 1. Resumo Executivo
A análise do dataset de logs do OpenStack identificou um volume significativo de operações normais pontuadas por sequências críticas de erros. De um total de **217.534 logs**, foram isoladas **1.204 anomalias** (0,55% do total). A investigação aprofundada revelou que os erros seguem padrões sequenciais claros, escalando de falhas de armazenamento para colapso de recursos do sistema.

### Métricas Principais
| Categoria | Quantidade |
| :--- | :--- |
| **Logs Normais** | 216.330 |
| **Logs de Erro (Anomalias)** | 1.204 |

---

## 2. Análise de Causa Raiz: 12 Tipos Específicos de Falha
Uma inspeção profunda dos templates de evento (`EventTemplate`) permitiu diagnosticar 12 tipos específicos de falhas, divididos em quatro áreas críticas:

### A. Infraestrutura e Sistema Operacional (Crítico)
1.  **Exaustão de Recursos (`Too many open files`):** O processo atingiu o limite de descritores de arquivo, causando falhas em cascata.
2.  **Erro de Resolução DNS:** Falha do serviço de nomes interno, impedindo comunicação entre nós.

### B. Rede e Conectividade
3.  **Rede Não Pronta (Provisionamento):** Tentativa de acesso à instância antes da conclusão da configuração do Neutron (HTTP 400).
4.  **Falha de Cache (Info Cache):** Dessincronização entre o banco de dados do Nova e o estado real da rede.

### C. Armazenamento (Storage)
5.  **Inconsistência de Volume:** "Race condition" ao tentar desanexar volumes já removidos.
6.  **Erro de Mapeamento (`KeyError: device_path`):** Falha do driver ao retornar o caminho do bloco do dispositivo.
7.  **Depreciação de API (Cinder v2):** Falhas por uso de endpoints legados.

### D. Falhas de Aplicação e Ciclo de Vida
8.  **Falha de Metadados (AttributeError):** Acesso a objetos nulos no serviço de metadados.
9.  **Erro de Lógica (TypeError):** Erro de programação no tratamento de respostas vazias.
10. **Falha de Busca:** O agendador não encontrou instâncias válidas para o evento.
11. **Falha de Reinicialização (Reboot/Shutdown):** Instâncias travadas que não respondem a sinais ACPI.
12. **Interrupção Genérica:** Logs de orquestração indicando falha total (`Failure!!!`).

---

## 3. Cronologia do Colapso
O incidente seguiu uma escalada progressiva:
1.  **Início:** Inconsistências de armazenamento e mapeamento de volumes.
2.  **Meio:** Degradação de rede (`FAILURE_SSH`) e aumento de latência.
3.  **Pico:** Saturação de recursos do SO (muitos arquivos abertos), derrubando DNS e Builds.
4.  **Final:** Falhas residuais de API e instâncias "zumbis".

---

## 4. Guia Técnico do Dataset
Para fins de reprodução e treinamento de modelos, é crucial entender a estrutura lógica deste arquivo.

### 4.1 A Natureza dos Dados
Este arquivo não é um log bruto aleatório, mas o resultado de uma **execução controlada de testes de carga e injeção de falhas**.
* **Estrutura:** Sequencial e cronológica.
* **Pré-processamento:** Logs já "parseados" (Template vs. Variáveis).
* **Ground Truth:** Rotulagem confiável de anomalias (`0` vs `1`).

### 4.2 Dicionário de Dados Essenciais
Para IA e AIOps, foque nestas 6 colunas críticas:

| Coluna | Descrição Técnica | Função na IA |
| :--- | :--- | :--- |
| **`timestamp`** | Data/hora precisa. | **Fundamental.** Define a causalidade temporal. |
| **`EventTemplate`** | Padrão fixo do log (ex: `Instance <*> created`). | **Feature Principal.** Transforma texto em categorias (IDs). |
| **`Content`** | Mensagem completa. | **Contexto.** Útil para embeddings semânticos (NLP). |
| **`anom_label`** | `0` (Normal) ou `1` (Erro). | **Target.** O alvo da previsão/classificação. |
| **`test_id`** | ID da sessão de teste. | **Agrupamento.** Define o escopo da sequência lógica. |
| **`service`** | Componente (Nova, Cinder, etc.). | **Filtro.** Ajuda a isolar a origem do problema. |

---

## 5. Protocolo de Experimentação (Machine Learning)
Recomenda-se a abordagem de **Next-Log Prediction** (Previsão do Próximo Log) para criar um sistema de alerta antecipado.

### 5.1 Preparação dos Dados (Split Cronológico)
Não use `random_split`. Respeite a linha do tempo.
* **Treino (8