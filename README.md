# 🏦 Risk Engine & Assessor de Investimentos com IA

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/status-AUC_ROC_Score_0.72-green)

## Sobre
Este projeto apresenta uma solução *end-to-end* para o mercado financeiro, desenvolvida com foco no ecossistema de uma corretora internacional.

A solução vai além da classificação binária tradicional (Paga/Não Paga). Ela integra um **Engine de Machine Learning** robusto para cálculo de probabilidade de *default* com um **Agente de Inteligência Artificial Generativa** que atua como um assessor financeiro, recomendando produtos de investimento personalizados com base no perfil de risco calculado.

---

## O Desafio e os Dados

### Seleção do Dataset
Utilizamos o dataset histórico do **Lending Club** (Empréstimos P2P), amplamente reconhecido na indústria de crédito por conter dados reais de comportamento financeiro, incluindo renda, DTI (*Debt-to-Income*), histórico de crédito e status de pagamento.

### Pré-processamento de Dados (Data Cleaning)
Para garantir a robustez do modelo, o pipeline de dados incluiu:
*   **Tratamento de Nulos:** Imputação estratégica baseada na distribuição das variáveis.
*   **Conversão de Tipos:** Tratamento de colunas categóricas e numéricas.
*   **Winsorização de Outliers:** Aplicação de cortes no percentil 99 (P99) para variáveis como *Renda Anual* e *DTI*, evitando que valores extremos distorcessem o gradiente do modelo.

### Engenharia de Atributos (Feature Engineering)
O diferencial de performance do modelo veio da criação de variáveis financeiras sintéticas que capturam a saúde do cliente melhor que os dados brutos:
*   `loan_to_income_ratio`: O peso do empréstimo sobre a renda anual.
*   `credit_history_length`: Tempo de exposição ao mercado de crédito.
*   `acc_open_rate`: Velocidade de abertura de novas contas (sinal de busca desesperada por crédito).

---

## Modelagem e Seleção (Model Selection)

Durante o desenvolvimento, testamos diferentes abordagens para maximizar a métrica alvo (**ROC-AUC**), dada a natureza desbalanceada de dados de fraude/calote.

### Benchmarking de Algoritmos
1.  **Regressão Logística:** Utilizada como *baseline*. Apresentou boa interpretabilidade, mas falhou em capturar relações não-lineares complexas entre *Juros* e *Risco*.
2.  **Random Forest:** Superou a Regressão Logística, mas apresentou sinais de *overfitting* e tempos de inferência mais altos.
3.  **XGBoost (Vencedor):** Escolhido como o motor final.
    *   **Motivo:** Melhor generalização em dados tabulares, tratamento nativo de valores nulos e velocidade de treino (`tree_method='hist'`).
    *   **Otimização:** Utilizamos `RandomizedSearchCV` com validação cruzada (3-fold) para tunar hiperparâmetros críticos como `learning_rate`, `max_depth` e `scale_pos_weight` (para lidar com o desbalanceamento de classes).

---

## Regra de Negócio e Segmentação

O modelo matemático entrega uma probabilidade (0 a 100%). Para tornar isso acionável para o negócio (Avenue), desenvolvemos uma régua de decisão baseada em apetite de risco:

| Faixa de Probabilidade | Classificação | Segmento | Ação Recomendada (Política de Crédito) |
| :--- | :--- | :--- | :--- |
| **0% - 15%** | Baixo Risco | **Prime** | Aprovação automática. Oferta de **Conta Margem**, Opções e ETFs de Ações. |
| **15% - 40%** | Médio Risco | **Standard** | Crédito sob análise. Foco em Renda Fixa (**Bonds**) e REITs para garantia. |
| **> 40%** | Alto Risco | **Restrito** | Crédito negado. Recomendação de preservação de patrimônio em **T-Bills** e Cash. |

---

## O Agente de IA Generativa (GenAI)

Para operacionalizar a decisão, integrei um módulo de **IA Generativa** (simulando a arquitetura do **Google Vertex AI / Gemini**).

*   **Input:** O Agente recebe o contexto técnico (Score do XGBoost + Top Fatores de Risco, ex: "DTI Alto").
*   **Persona:** Atua como um Assessor Sênior de Investimentos.
*   **Output:** Gera um pitch comercial personalizado, explicando ao cliente o porquê da decisão e sugerindo os produtos da matriz de recomendação acima.

> *Nota: No notebook, o agente opera com uma lógica de fallback para garantir a execução sem necessidade de chaves de API ativas.*

---

## Resultados Finais

O modelo final atingiu métricas sólidas para a concessão de crédito:

*   **AUC-ROC:** **0.7256** (Excelente capacidade de ordenação de risco).
*   **Recall (Inadimplentes):** **67%** (O modelo identifica 2 em cada 3 possíveis calotes, protegendo o caixa da empresa).
*   **Impacto:** A segmentação isolou os **6% melhores clientes** para estratégias de retenção agressiva, enquanto protege a exposição da empresa nos 60% da base de maior risco.

---

---

## 🚀 Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/umbura/Risk-Engine-AI-Investment-Advisor.git
    ```
2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute o Notebook:**
    Abra o arquivo `risk_engine_xgboost.ipynb`.

---
