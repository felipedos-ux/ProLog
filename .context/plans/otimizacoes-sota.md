---
status: unfilled
generated: 2026-03-15
agents:
  - type: "code-reviewer"
    role: "Review code changes for quality, style, and best practices"
  - type: "bug-fixer"
    role: "Analyze bug reports and error messages"
  - type: "feature-developer"
    role: "Implement new features according to specifications"
  - type: "refactoring-specialist"
    role: "Identify code smells and improvement opportunities"
  - type: "test-writer"
    role: "Write comprehensive unit and integration tests"
  - type: "documentation-writer"
    role: "Create clear, comprehensive documentation"
  - type: "performance-optimizer"
    role: "Identify performance bottlenecks"
  - type: "security-auditor"
    role: "Identify security vulnerabilities"
  - type: "backend-specialist"
    role: "Design and implement server-side architecture"
  - type: "frontend-specialist"
    role: "Design and implement user interfaces"
  - type: "architect-specialist"
    role: "Design overall system architecture and patterns"
  - type: "devops-specialist"
    role: "Design and maintain CI/CD pipelines"
  - type: "database-specialist"
    role: "Design and optimize database schemas"
  - type: "mobile-specialist"
    role: "Develop native and cross-platform mobile applications"
docs:
  - "project-overview.md"
  - "architecture.md"
  - "development-workflow.md"
  - "testing-strategy.md"
  - "glossary.md"
  - "data-flow.md"
  - "security.md"
  - "tooling.md"
phases:
  - id: "phase-1"
    name: "Architectural Planning & Data Transformation"
    prevc: "P"
    agent: "architect-specialist"
  - id: "phase-2"
    name: "Core Features Implementation (Thresholds & Loss)"
    prevc: "E"
    agent: "feature-developer"
  - id: "phase-3"
    name: "Advanced Training (Stages & Contrastive)"
    prevc: "E"
    agent: "performance-optimizer"
  - id: "phase-4"
    name: "Validation & Benchmarking"
    prevc: "V"
    agent: "test-writer"
---

# Implementação Otimizações SOTA (Parser-Free) Plan

> Implementar as técnicas do estado da arte (Regex Mínimo, Threshold Adaptativo, Contrastive Learning) levantadas no Relatório de Implementação para evoluir as métricas do HDFS model (F1 de 0.86 para 0.95+).

## Task Snapshot
- **Primary goal:** Elevar o F1-Score do modelo LogGPT BPE de 86-88% para atingir a zona de SOTA (95-98%) utilizando técnicas estritamente não-discretizadas (Parser-Free).
- **Success signal:** Após a implementação do pipeline inteiro, o script de avaliação registrará F1 > 0.95 no dataset HDFS.
- **Key references:**
  - `docs/RELATORIO_IMPLEMENTACAO_OTIMIZACOES.md`

## Worker Lineup
- **`architect-specialist`**: Desenha a transformação do pre-processamento.
- **`feature-developer`**: Codifica as novas regras dinâmicas de Loss e a janela de contexto longa (Block Size).
- **`performance-optimizer`**: Implementa o fluxo de treinamento contrastivo pesado e a fase de *Domain Adaptation*.
- **`test-writer`**: Cria os benchmarks finais que gerarão as tabelas para o TCC.

---

## Working Phases

### Phase 1 — Architectural Planning & Data Transformation
> **Primary Agent:** `architect-specialist`

**Objective:** Implementar o Regex Mínimo (inspirado no LogLLM) para reduzir a entropia visual do dataset sem usar parsers predatórios como o Drain.

**Tasks**

| # | Task | Agent | Status | Deliverable |
|---|------|-------|--------|-------------|
| 1.1 | Criar função de limpeza que traduz IPs, HEX e Timestamps dinâmicos para tokens `<IP>`, `<HEX>` em `preprocessing.py` | `feature-developer` | done | `preprocessing.py` modificado |
| 1.2 | Refatorar os pipelines Dataloader para ler o dataframe processado no fluxo do modelo | `architect-specialist` | done | Modificações no `TrainLoader` |
| 1.3 | **Avaliar Métrica Cumulativa (Passo 1: Regex)**: Rodar o pipeline atualizado com Regex Mínimo e metrificar o F1 | `test-writer` | done | **F1: 0.8769** (P: 0.9613, R: 0.8062) |

---

### Phase 2 — Core Features Implementation (Thresholds & Loss)
> **Primary Agent:** `feature-developer`

**Objective:** Implementar Janela de Contexto Longa e Thresholds Adaptativos baseados em Desvio Padrão (inspirado no LAnoBERT e ADALog).

**Tasks**

| # | Task | Agent | Status | Deliverable |
|---|------|-------|--------|-------------|
| 2.1 | Aumentar o hyperparameter _Block Size_ ou implementar overlap context | `feature-developer` | done | Ajustes no `config.py` |
| 2.2 | Criar loop `detect_dynamic.py` que infere logs normais, tira Media/Std Dev da perplexidade, e usa como threshold | `feature-developer` | done | Threshold adaptativo em `detect_threshold.py` |
| 2.3 | **Avaliar Métrica Cumulativa (Passo 2: Regex + Adaptativo)**: Rodar a avaliação somando o Threshold Adaptativo sobre os dados limpos por Regex e metrificar o F1 | `test-writer` | done | **F1: 0.8712** (P: 0.9686, R: 0.7915) |

---

### Phase 3 — Advanced Training (Contrastive Learning)
> **Primary Agent:** `performance-optimizer`

**Objective:** Injetar técnicas que mudam o pre-treinamento visando robustez contínua (inspirado no ContraLog), utilizando **exclusivamente os dados do HDFS**. Faremos isso através de Self-Supervised Contrastive Learning com Augmentation.

**Tasks**

| # | Task | Agent | Status | Deliverable |
|---|------|-------|--------|-------------|
| 3.1 | Desenvolver Data Augmentation para logs (random masking, dropping) focado em criar visões (pares positivos) | `feature-developer` | done | `dataset_contrastive.py` modificado |
| 3.2 | Injetar InfoNCE Loss (Contrastive Learning) simulando representações robustas para anomalias, em conjunto à Cross-Entropy | `performance-optimizer` | done | `train_contrastive.py` |
| 3.3 | **Avaliar Métrica Cumulativa (Passo 3: Regex + Adaptativo + Contrastive)**: Rodar o motor contrastivo com HDFS 5k puro e avaliar salto do F1 | `test-writer` | pending | Tabela Comparativa Atualizada |

---

### Phase 4 — Final Joint Benchmarking
> **Primary Agent:** `test-writer`

**Objective:** Averiguar as métricas combinadas (Cumulative Study) de forma isenta e registrar o F1 Score absoluto com todas as técnicas SOTA rodando simultaneamente.

**Tasks**

| # | Task | Agent | Status | Deliverable |
|---|------|-------|--------|-------------|
| 4.1 | **Avaliação Sistêmica Final**: Confirmar o pipeline final inteiro (Regex + Adaptive + Contrastive) no Test Set | `test-writer` | pending | Métrica de Benchmarking Global (Meta: > 95%) |
| 4.2 | Exportar Confusion Matrix final das contribuições conjuntas e atualizar os resultados SOTA | `documentation-writer` | pending | Matriz Documentada |
