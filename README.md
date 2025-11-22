# Risk Engine & Assessor de Investimentos com IA(Project Nemesis)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Status](https://img.shields.io/badge/status-AUC_ROC_Score_0.72-green)

## Sobre
O projeto Nemesis apresenta uma solução completa para avaliação de risco de crédito e recomendação de investimentos, alinhada ao contexto operacional de uma corretora internacional. 

A abordagem combina um motor de Machine Learning para estimar probabilidade de risco de inadimplência com um agente de IA Gen capaz de produzir recomendações personalizadas com base no perfil de risco do cliente.

A solução integra engenharia de dados, modelagem preditiva, definição de políticas de crédito e geração de recomendações acionáveis para o usuário final.

> *Nota: Sem sombra de dúvidas, Nemesis foi o projeto mais difícil que desenvolvi até então. Precisei aplicar meus conhecimentos tanto em SQL quanto em Python, e mesmo assim obtive um resultado que considero mediano.*
> *Aprendi bastante, mas ao mesmo tempo senti frustração devido às limitações técnicas. Irei detalhar mais sobre isso ao longo deste documento.*

---

## O Desafio e os Dados

### Seleção do Dataset
Utilizamos o dataset histórico do **Lending Club** (Empréstimos P2P), amplamente reconhecido na indústria de crédito por conter 10 anos de dados reais de comportamento financeiro, incluindo renda, DTI (*Debt-to-Income*), histórico de crédito e status de pagamento.

### Pré-processamento de Dados (Data Cleaning)
Para garantir a robustez do modelo, o pipeline de dados incluiu:
*   **Tratamento de Nulos:** Imputação estratégica baseada na distribuição das variáveis.
*   **Conversão de Tipos:** Tratamento de colunas categóricas e numéricas.
*   **Winsorização de Outliers:** Aplicação de cortes no percentil 99 (P99) para variáveis como *Renda Anual* e *DTI*, evitando que valores extremos distorcessem o gradiente do modelo.

### Engenharia de Atributos (Feature Engineering)
O diferencial de performance do modelo veio da criação de variáveis financeiras sintéticas que capturam a saúde do cliente melhor que os dados brutos:
*   `loan_to_income_ratio`: proporção do valor do empréstimo em relação à renda.
*   `credit_history_length`: tempo de histórico de crédito ativo.
*   `acc_open_rate`: Velocidade de abertura de novas contas (sinal de busca desesperada por crédito).

> *Nota: Optei por utilizar o GCP porque queria aprender a trabalhar com o BigQuery. Até então, minha experiência com SQL se limitava a SQL Server e Postgres.*
>
> *Tive alguns problemas inicialmente, principalmente pelo fato de o dataset ser muito grande. Havia muitos dados a serem processados, e minhas primeiras tentativas de fazer o upload falharam. Foi então que decidi cruzar apenas os dados essenciais na etapa de processamento, o que não só aumentou o score final como também reduziu significativamente o tamanho do dataset, tornando-o mais fácil de manipular.*

---

## Modelagem e Seleção (Model Selection)

Durante o desenvolvimento, testamos diferentes abordagens para maximizar a métrica alvo (**ROC-AUC**), dada a natureza desbalanceada de dados de fraude/calote.

### Benchmarking de Algoritmos
1.  **Regressão Logística:** Utilizada como *baseline*. Apresentou boa interpretabilidade, mas falhou em capturar relações não-lineares complexas entre *Juros* e *Risco*.
2.  **Random Forest:** Superou a Regressão Logística, mas apresentou sinais de *overfitting* e tempos de inferência mais altos.
3.  **XGBoost (Vencedor):** Escolhido como o motor final.
    *   **Motivo:** Melhor generalização em dados tabulares, tratamento nativo de valores nulos e velocidade de treino (`tree_method='hist'`).
    *   **Otimização:** Utilizamos `RandomizedSearchCV` com validação cruzada (3-fold) para tunar hiperparâmetros críticos como `learning_rate`, `max_depth` e `scale_pos_weight` (para lidar com o desbalanceamento de classes).

> *Nota: Também foi testado o método de Stacking, mas ele se mostrou inferior ao XGBoost e apresentou indícios de overfitting.*

---

## Regra de Negócio e Segmentação

O modelo matemático entrega uma probabilidade (0 a 100%). Para tornar isso acionável para o negócio, desenvolvemos uma régua de decisão baseada em apetite de risco:

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

> *Nota: Devido a limitações técnicas, não consegui aplicar um agente real, então tive de simulá-lo. No notebook, o agente opera com uma lógica de fallback para garantir a execução sem necessidade de chaves de API ativas..*

---

## Resultados Finais

O modelo final atingiu métricas sólidas para a concessão de crédito:

*   **AUC-ROC:** **0.7256** indica boa capacidade de ordenação de risco.
*   **Recall (Inadimplentes):** **67%** significa que o modelo identifica 2 em cada 3 possíveis calotes, protegendo o caixa da empresa.
*   **Impacto:** A segmentação isolou os **6% melhores clientes** para estratégias de retenção agressiva, enquanto protege a exposição da empresa nos 60% da base de maior risco.

---

## Considerações Finais
*Pelo que estudei, modelos de risco são mais difíceis de alcançar scores altos do que os modelos habituais, justamente devido à complexidade das operações financeiras. Minha meta era alcançar um score de 0,80, mas infelizmente não foi possível. No futuro, pretendo relançar este modelo tentando atingir essa meta.*

---

## Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/umbura/Risk-Engine-AI-Investment-Advisor.git
    ```
2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Extraia o dataset:**
   Extraia o dataset processado `arquivo lending_club_loans_processed.rar`.
    
4.  **Execute o Notebook:**
    Abra o arquivo `risk_engine_xgboost.ipynb`.

---

## Créditos e Referência

**All Lending Club loan data**.[All Lending Club Loan Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club/data)

---

## Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
