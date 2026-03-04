# TriaLogic 🏥🤖

**TriaLogic** é um framework multi-agente para triagem clínica automatizada, desenvolvido como Trabalho de Conclusão de Curso. Utiliza uma arquitetura *Agentic Workflow* orquestrada por **LangGraph** para extrair sinais vitais de notas clínicas não estruturadas (MIMIC-IV-Notes), validar os dados fisiologicamente, calcular escores de risco (NEWS2 e MEWS), e produzir recomendações clínicas ancoradas em evidências via Retrieval-Augmented Generation (RAG) — tudo executado localmente com **LLaMA 3.1 8B** via **Ollama**, sem dependência de APIs externas.

## 🚀 Características Principais

*   **Orquestração Multi-Agente**: Powered by **LangGraph**, com um Supervisor que gerencia estado compartilhado e roteia tarefas condicionalmente.
*   **Extração Clínica (Scribe)**: Transforma texto clínico não estruturado em dados estruturados via esquemas Pydantic (SBP, DBP, HR, RR, SpO₂, Temperatura, ACVPU).
*   **Validação Fisiológica (Validator)**: Verifica plausibilidade fisiológica com regras determinísticas, conversão automática Fahrenheit→Celsius e loop de retry (até 3 tentativas).
*   **Análise de Risco (Mathematician)**: Calcula escores NEWS2 e MEWS usando *tool calling* determinístico (padrão) ou LLM-based probabilístico (configurável).
*   **Recuperação de Evidências (Clinical RAG)**: Recupera diretrizes clínicas (Sepsis-3, NEWS2, MEWS, protocolos de trauma/cardiologia/pneumologia) de um banco vetorial ChromaDB.
*   **Síntese e Auditoria (Synthesizer)**: Produz relatório final com verificação de fidelidade de citação (*quote fidelity check*) como mecanismo anti-alucinação.
*   **Processamento em Lote**: Processamento de datasets completos com retomada automática, medição de latência por caso e relatório estatístico.

## 🏗️ Arquitetura

O sistema segue um workflow de grafo dirigido com roteamento condicional:

1.  **Input**: Nota clínica não estruturada (sumário de alta do MIMIC-IV-Notes).
2.  **Supervisor (Planning + Router)**: Planeja a sequência de agentes e roteia condicionalmente com base no estado.
3.  **Scribe Agent**: Extrai sinais vitais em formato estruturado (Pydantic `ClinicalSchema`).
4.  **Validator Node**: Verifica ranges fisiológicos, converte unidades e realiza *scrubbing* de valores inválidos.
5.  **Mathematician Agent**: Computa NEWS2 e MEWS via *tool calling* determinístico (`calculate_clinical_score`).
6.  **Clinical RAG Agent**: Recupera diretrizes clínicas relevantes do banco vetorial ChromaDB.
7.  **Synthesizer Agent**: Sintetiza relatório de auditoria final com extração de evidências e verificação de fidelidade.

```mermaid
stateDiagram-v2
    direction TB

    %% --- Estilos ---
    classDef mainNode fill:#ececff,stroke:#333,stroke-width:2px;
    classDef errorNode fill:#ffcccc,stroke:#b30000,stroke-width:2px,stroke-dasharray: 5 5;
    classDef finalNode fill:#ccffcc,stroke:#006600,stroke-width:2px;

    %% --- Nós Principais ---
    state "Início (Raw Clinical Text)" as Start
    state "Fim (Structured JSONL Output)" as End
    
    state "🧠 Supervisor Router" as Supervisor
    note right of Supervisor
        Hub Central: Gerencia Estado Global
        Roteia baseado na presença de dados
        extraídos, escores calculados, etc.
    end note

    %% --- FASE 1: Extração e Auto-Correção ---
    state "Fase 1: Extração & Validação" as Phase1 {
        direction LR
        state "✍️ Scribe Agent" as Scribe
        note right of Scribe
            Extração de sinais vitais
            (BP, HR, RR, SpO2, Temp)
            com truncagem inteligente
        end note

        state "🛡️ Validator Node" as Validator
        
        state "Loop de Retry" as RetryChoice <<choice>>

        Scribe --> Validator: JSON Estruturado
        Validator --> RetryChoice: Validação de Schema
        RetryChoice --> Scribe: ❌ Erro (Feedback Injetado)
    }

    %% --- FASE 2: Cálculo de Escores de Risco ---
    state "Fase 2: Análise de Risco" as Phase2 {
        state "🧮 Mathematician Agent" as Math
        note right of Math
            Cálculo de NEWS2/MEWS
            Tool Calling: Python puro
            Suporte probabilístico
        end note
    }

    %% --- FASE 3: Auditoria Clínica com RAG ---
    state "Fase 3: Auditoria & RAG" as Phase3 {
        direction TB
        state "🔍 Clinical RAG" as RAG
        state "⚖️ Synthesizer Agent" as Synthesizer
        
        RAG --> Synthesizer: Contexto Clínico + Diretrizes
        note right of Synthesizer
            Auditoria baseada em evidências
            Quote Fidelity Check
            (Anti-alucinação)
        end note
    }

    %% --- Conexões do Workflow ---
    Start --> Supervisor

    %% Roteamento Condicional do Supervisor
    Supervisor --> Scribe: 1. Extrair Sinais Vitais
    RetryChoice --> Supervisor: ✅ Dados Válidos
    
    Supervisor --> Math: 2. Calcular Escores (NEWS2/MEWS)
    Math --> Supervisor: Score Report Gerado
    
    Supervisor --> RAG: 3. Buscar Protocolos Clínicos
    Synthesizer --> End: Relatório Final de Auditoria

    %% Aplicação de Classes CSS
    class Supervisor mainNode
    class End finalNode
    class RetryChoice errorNode
```

## 📂 Estrutura do Projeto

```text
TriaLogic/
├── chroma_db/                  # Banco de dados vetorial (ChromaDB) para RAG
├── data/                       # Datasets
│   ├── discharge.csv           # Notas de alta originais (MIMIC-IV-Notes)
│   ├── discharge_filtered.csv  # Notas filtradas por coorte
│   ├── gold_standard_dataset.csv  # Dataset de avaliação (240 casos)
│   ├── ground_truth.csv        # Ground truth com sinais vitais anotados (154 hadm_ids)
│   ├── master_dataset.csv      # Dataset consolidado
│   └── validation_notes.csv    # Notas para validação
├── docs/                       # Diretrizes clínicas para RAG
│   ├── definitions.txt         # Definições clínicas (NEWS2, MEWS)
│   ├── Sepsis-3.pdf            # Protocolo Sepsis-3
│   ├── NEWS2_Chart.pdf         # Tabela NEWS2
│   ├── Cardio.pdf / ChestPain.pdf / Trauma.pdf / ...  # Protocolos clínicos
│   └── ...
├── prompts/                    # Prompts de sistema para cada agente
│   ├── scribe_prompt.md        # Extração de sinais vitais
│   ├── mathematician_prompt.md # Cálculo e interpretação de escores
│   ├── rag_prompt.md           # Geração de query RAG
│   ├── auditor_prompt.md       # Síntese e auditoria clínica
│   └── one_shot_prompt.md      # Prompt one-shot (baseline)
├── results/                    # Resultados experimentais
│   ├── experiment_results_v1.jsonl        # TriaLogic (Agents)
│   ├── norag_experiment_results_v1.jsonl  # TriaLogic sem RAG
│   ├── novalidation_experiment_results_v1.jsonl  # TriaLogic sem Validator
│   ├── probabilistic_experiment_results_v1.jsonl # TriaLogic Probabilístico
│   ├── baseline_results.jsonl   # Baseline Zero-Shot
│   ├── oneshot_baseline_results.jsonl  # Baseline One-Shot
│   ├── *_latency.json          # Sumários de latência por configuração
│   ├── tcc_final_metrics.*     # Métricas finais (CSV, LaTeX, HTML, Markdown)
│   ├── tcc_significance.md     # Intervalos de confiança e testes de significância
│   └── per_system/             # Relatórios HTML coloridos por sistema
├── scripts/                    # Scripts de execução e avaliação
│   ├── run_batch_processing.py # Processamento em lote (configurável via flags)
│   ├── run_baseline.py         # Baseline Zero-Shot
│   ├── run_baseline_oneshot.py # Baseline One-Shot
│   ├── run_single.py           # Análise de caso único
│   ├── evaluation.py           # Avaliação com métricas, bootstrap CI, McNemar
│   ├── ingest_knowledge.py     # Ingestão de PDFs no ChromaDB
│   ├── filter_dataset.py       # Filtragem e amostragem do dataset
│   └── plot_risk_distribution.py  # Visualização de distribuições
├── src/                        # Código-fonte principal
│   ├── main.py                 # Definição e compilação do grafo LangGraph
│   ├── agents/                 # Implementação dos agentes
│   │   ├── supervisor.py       # Planejamento e roteamento condicional
│   │   ├── scribe.py           # Extração clínica estruturada
│   │   ├── validator.py        # Validação fisiológica com retry loop
│   │   ├── mathematician.py    # Cálculo de escores (determinístico/probabilístico)
│   │   ├── clinical_rag.py     # Recuperação de evidências clínicas
│   │   └── synthesizer.py      # Síntese e auditoria final
│   ├── schemas/                # Modelos Pydantic
│   │   ├── scribe_schema.py    # ClinicalSchema (vitais + ACVPU)
│   │   ├── mathematician_schema.py  # MathematicianSchema (escores)
│   │   ├── auditor_schema.py   # AuditorOutput (relatório)
│   │   └── input_schema.py     # InputSchema (entrada do pipeline)
│   ├── state/                  # Estado compartilhado do LangGraph
│   │   └── agent_state.py      # AgentState (TypedDict)
│   ├── tools/                  # Ferramentas de tool calling
│   │   └── calculator.py       # Calculadora determinística NEWS2/MEWS
│   └── utils/                  # Utilitários
│       ├── vectorstore.py      # Singleton ChromaDB com cache
│       ├── check_quote.py      # Verificação de fidelidade de citação
│       ├── vitals_normalizer.py # Normalização de sinais vitais
│       └── run_with_timeout.py # Timeout para chamadas LLM
└── requirements.txt            # Dependências
```

## 🛠️ Instalação

1. **Clone o repositório**:

    ```bash
    git clone https://github.com/yourusername/TriaLogic.git
    cd TriaLogic
    ```

2. **Configure o ambiente Python** (Python 3.11+):

    ```bash
    # Opção 1: Usando Conda
    conda create -n trialogic python=3.11
    conda activate trialogic

    # Opção 2: Usando venv
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3. **Instale as dependências**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Instale o Ollama e baixe o modelo** (pré-requisito):

    Instale o [Ollama](https://ollama.ai) e baixe o modelo LLaMA 3.1 8B:

    ```bash
    ollama pull llama3.1:8b
    ```

5. **Prepare a Base de Conhecimento** (RAG):

    ```bash
    python -m scripts.ingest_knowledge
    ```

## 🏃 Uso

### Processamento em Lote (Experimento Principal)

```bash
# TriaLogic completo (Validator + RAG + Determinístico)
python -m scripts.run_batch_processing -o results/experiment_results.jsonl

# Sem Validator
python -m scripts.run_batch_processing --no-validator -o results/novalidation_results.jsonl

# Sem RAG
python -m scripts.run_batch_processing --no-rag -o results/norag_results.jsonl

# Mathematician Probabilístico (LLM-based)
python -m scripts.run_batch_processing --probabilistic -o results/probabilistic_results.jsonl
```

### Baselines

```bash
# Baseline Zero-Shot
python -m scripts.run_baseline -o results/baseline_b0.jsonl

# Baseline One-Shot
python -m scripts.run_baseline_oneshot -o results/baseline_b1.jsonl
```

### Avaliação e Métricas

```bash
python -m scripts.evaluation
```

Gera automaticamente: CSV, LaTeX, HTML, Markdown, relatórios por sistema, intervalos de confiança (bootstrap) e teste de McNemar.

### Caso Único (Teste)

```bash
python -m scripts.run_single
```

## 📊 Resultados

### Métricas Agregadas por Sistema

| Sistema                   | MAE Médio | Taxa Alucinação | Macro F1 (IC 95%)         | N Amostras |
|---------------------------|-----------|-----------------|---------------------------|------------|
| Baseline (Zero-Shot)      | 3.66      | 11.95%          | 0.793 [0.765, 0.820]      | 1088       |
| Baseline (One-Shot)       | 2.85      | 4.80%           | 0.843 [0.817, 0.867]      | 1084       |
| No RAG                    | 2.55      | 2.78%           | 0.863 [0.838, 0.888]      | 1007       |
| **TriaLogic (Agents)**    | **2.38**  | **2.60%**       | **0.875 [0.851, 0.898]**  | 1001       |
| TriaLogic (No Validator)  | 10.15     | 17.08%          | 0.723 [0.689, 0.755]      | 1001       |
| TriaLogic (Probabilistic) | 2.79      | 8.81%           | 0.875 [0.852, 0.898]      | 987        |

### Significância Estatística

* **McNemar TA vs B0 (Extração)**: n=742 pares, b=101 vs c=43, p < 0.001
* **McNemar TA vs B0 (NEWS)**: n=129 pares, b=48 vs c=10, p < 0.001

### Latência por Caso

| Sistema              | N    | Mediana (s) | Média (s) | DP (s) | P95 (s) |
|----------------------|------|-------------|-----------|--------|---------|
| Baseline (Zero-Shot) | 104  | 20.56       | 28.66     | 23.74  | 70.92   |
| Baseline (One-Shot)  | 77   | 46.38       | 66.22     | 45.37  | 167.77  |
| TriaLogic (Agents)   | 23   | 151.98      | 151.74    | 25.58  | 187.28  |

*Medições em hardware local: AMD Ryzen 5 5500H, 16 GB RAM, NVIDIA RTX 3050 4 GB VRAM, Ollama 0.17.0.*

## 🧠 Agentes

* **Supervisor**: Planejamento e roteamento condicional baseado em estado. Factory pattern parametrizado por `use_rag` e `use_validator`.
* **Scribe**: Extração de sinais vitais em dois passos (identificação de span + extração estruturada). Saída validada via `ClinicalSchema` (Pydantic).
* **Validator**: Verificação de plausibilidade fisiológica com regras determinísticas, conversão automática Fahrenheit→Celsius, e *scrubbing* compulsório após 3 tentativas falhadas.
* **Mathematician**: Cálculo de NEWS2 e MEWS via *tool calling* determinístico (`calculator.py`). Modo probabilístico (LLM-based) disponível como configuração alternativa.
* **Clinical RAG**: Geração de query contextual + recuperação de diretrizes clínicas do ChromaDB (embeddings `all-MiniLM-L6-v2`).
* **Synthesizer**: Síntese de relatório final com extração de evidências do contexto RAG, *quote fidelity check* e limiar de similaridade mínima (0.45).

### Stack Tecnológica

* **LangGraph**: Orquestração de agentes (grafo de estados com roteamento condicional)
* **LLaMA 3.1 8B (via Ollama)**: LLM local (temperatura=0, seed=42, num_ctx=8192)
* **HuggingFace Sentence-Transformers**: Embeddings para RAG (`all-MiniLM-L6-v2`)
* **ChromaDB**: Banco de dados vetorial para RAG
* **Pydantic**: Validação e estruturação de esquemas de dados
* **LangChain**: Integração com LLM, prompts e ferramentas
* **scikit-learn**: Métricas de avaliação (precision, recall, F1)

## ⚠️ Limitações

* **Inferência local**: Latência ~2.5 min/caso em hardware com GPU de 4 GB VRAM (offloading parcial CPU/GPU). Servidores com GPUs dedicadas (24+ GB VRAM) reduziriam significativamente.
* **Parsing Success Rate**: 80.4% nas configurações com agentes vs 89.0% no baseline, devido ao rigor dos esquemas Pydantic e validação fisiológica.
* **Modelo 8B**: Raciocínio aritmético limitado — o modo probabilístico apresenta 56.9% de alucinação no NEWS vs 4.7% no modo determinístico.
* **Dataset**: 154 admissões únicas do MIMIC-IV-Notes (Beth Israel Deaconess Medical Center). Variabilidade geográfica e institucional não coberta.
* **Avaliação RAG**: Apenas qualitativa — ausência de ground truth de condutas médicas para métricas automatizadas.
* **Janela de contexto**: Limitada a 8.192 tokens, impactando notas clínicas atipicamente longas.

## 📚 Referências Clínicas

* **NEWS2**: National Early Warning Score 2 (Royal College of Physicians)
* **MEWS**: Modified Early Warning Score
* **Sepsis-3**: Third International Consensus Definitions for Sepsis
* **MIMIC-IV**: Medical Information Mart for Intensive Care (PhysioNet)

## 📄 Licença

[MIT License](LICENSE.md)

---

**Aviso**: Este sistema é destinado apenas para fins de pesquisa e educação. Não deve ser usado para tomada de decisões clínicas sem validação e supervisão médica adequadas.
