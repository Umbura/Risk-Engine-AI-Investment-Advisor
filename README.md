<div align="center">

# Risk Engine & AI Investment Advisor

### Project Nemesis

<!-- LANGUAGE SWITCHER -->
[![Read in Portuguese](https://img.shields.io/badge/Read%20in-Portuguese-2ea44f?style=for-the-badge&logo=google-translate&logoColor=white)](README_PT.md)

<!-- TECH STACK BADGES -->
<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Model-XGBoost-green" alt="XGBoost">
  <img src="https://img.shields.io/badge/status-AUC_ROC_Score_0.72-green" alt="Status AUC">
</p>

<!-- MAIN IMAGE -->
<img src="assets/https://raw.githubusercontent.com/Umbura/Risk-Engine-AI-Investment-Advisor/refs/heads/main/assets/classification_report.png" alt="App Flowchart" width="100%">

*(--- Relatório de Classificação ---
              precision    recall  f1-score   support

        Pago       0.89      0.66      0.76     20167
Inadimplente       0.33      0.67      0.44      5006

    accuracy                           0.66     25173
   macro avg       0.61      0.67      0.60     25173
weighted avg       0.78      0.66      0.70     25173

AUC-ROC Final (Teste): 0.7256)*

</div>

---

## About
Project Nemesis presents a complete solution for credit risk assessment and investment recommendation, aligned with the operational context of an international brokerage.

The approach combines a Machine Learning engine to estimate default risk probability with a Generative AI agent capable of producing personalized recommendations based on the client's risk profile.

The solution integrates data engineering, predictive modeling, credit policy definition, and actionable recommendation generation for the end user.

> *Note: Without a doubt, Nemesis was the most difficult project I have developed so far. I had to apply my knowledge in both SQL and Python, and even then, I obtained a result that I consider average.*
> *I learned a lot, but at the same time felt frustration due to technical limitations. I will detail more about this throughout this document.*

---

## The Challenge and Data

### Dataset Selection
We used the historical **Lending Club** dataset (P2P Loans), widely recognized in the credit industry for containing 10 years of real financial behavior data, including income, DTI (*Debt-to-Income*), credit history, and payment status.

### Data Cleaning (Pre-processing)
To ensure model robustness, the data pipeline included:
*   **Null Treatment:** Strategic imputation based on variable distribution.
*   **Type Conversion:** Handling of categorical and numerical columns.
*   **Outlier Winsorization:** Application of cuts at the 99th percentile (P99) for variables such as *Annual Income* and *DTI*, preventing extreme values from distorting the model gradient.

### Feature Engineering
The performance differential came from creating synthetic financial variables that capture client health better than raw data:
*   `loan_to_income_ratio`: Proportion of the loan amount relative to income.
*   `credit_history_length`: Duration of active credit history.
*   `acc_open_rate`: Speed of opening new accounts (a signal of desperate credit-seeking behavior).

> *Note: I chose to use GCP because I wanted to learn how to work with BigQuery. Until then, my SQL experience was limited to SQL Server and Postgres.*
>
> *I had some issues initially, mainly because the dataset was very large. There was a lot of data to process, and my first attempts to upload failed. It was then that I decided to cross-reference only the essential data during the processing stage, which not only increased the final score but also significantly reduced the dataset size, making it easier to manipulate.*

---

## Model Selection

During development, we tested different approaches to maximize the target metric (**ROC-AUC**), given the imbalanced nature of fraud/default data.

### Algorithm Benchmarking
1.  **Logistic Regression:** Used as a *baseline*. It showed good interpretability but failed to capture complex non-linear relationships between *Interest Rates* and *Risk*.
2.  **Random Forest:** Outperformed Logistic Regression but showed signs of *overfitting* and higher inference times.
3.  **XGBoost (Winner):** Chosen as the final engine.
    *   **Reason:** Better generalization on tabular data, native handling of null values, and training speed (`tree_method='hist'`).
    *   **Optimization:** Used `RandomizedSearchCV` with cross-validation (3-fold) to tune critical hyperparameters like `learning_rate`, `max_depth`, and `scale_pos_weight` (to handle class imbalance).

> *Note: Stacking was also tested, but it proved inferior to XGBoost and showed signs of overfitting.*

---

## Business Rules & Segmentation

The mathematical model delivers a probability (0 to 100%). To make this actionable for the business, we developed a decision ruler based on risk appetite:

| Probability Range | Classification | Segment | Recommended Action (Credit Policy) |
| :--- | :--- | :--- | :--- |
| **0% - 15%** | Low Risk | **Prime** | Automatic approval. Offer **Margin Account**, Options, and Stock ETFs. |
| **15% - 40%** | Medium Risk | **Standard** | Credit under analysis. Focus on Fixed Income (**Bonds**) and REITs for collateral. |
| **> 40%** | High Risk | **Restricted** | Credit denied. Recommendation for wealth preservation in **T-Bills** and Cash. |

---

## The Generative AI Agent (GenAI)

To operationalize the decision, I integrated a **Generative AI** module (simulating the **Google Vertex AI / Gemini** architecture).

*   **Input:** The Agent receives the technical context (XGBoost Score + Top Risk Factors, e.g., "High DTI").
*   **Persona:** Acts as a Senior Investment Advisor.
*   **Output:** Generates a personalized commercial pitch, explaining the decision to the client and suggesting products from the recommendation matrix above.

> *Note: Due to technical limitations, I could not apply a real agent, so I had to simulate it. In the notebook, the agent operates with fallback logic to ensure execution without the need for active API keys.*

---

## Final Results

The final model achieved solid metrics for credit granting:

*   **AUC-ROC:** **0.7256** indicates good risk ordering capability.
*   **Recall (Defaulters):** **67%** means the model identifies 2 out of every 3 potential defaults, protecting the company's cash flow.
*   **Impact:** Segmentation isolated the **top 6% best clients** for aggressive retention strategies, while protecting the company's exposure in the 60% highest-risk base.

---

## Final Considerations
*From what I studied, risk models are harder to achieve high scores on compared to typical models, precisely due to the complexity of financial operations. My goal was to reach a score of 0.80, but unfortunately, it was not possible. In the future, I plan to relaunch this model attempting to reach that goal.*

---

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/umbura/Risk-Engine-AI-Investment-Advisor.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Extract the dataset:**
    Extract the processed dataset `lending_club_loans_processed.rar`.
    
4.  **Run the Notebook:**
    Open the file `risk_engine_xgboost.ipynb`.

---

## Credits and References

**All Lending Club loan data**. [All Lending Club Loan Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club/data)

---

## License

Distributed under the MIT license. See the `LICENSE` file for more details.
