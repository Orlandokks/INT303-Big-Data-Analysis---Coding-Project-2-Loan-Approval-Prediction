# INT303 – Coursework 2  
## Loan Approval Prediction for a FinTech Lender

> **Goal:** Build an end-to-end machine learning pipeline that predicts **loan approval probability** and turns it into **actionable decision rules** (approve / decline / human review), while keeping **risk, transparency, and ethics** in view.

---

## Why this project matters

Loan approval is not just a classification task — it’s a high-impact decision system.

In this coursework, I treated the problem like a real FinTech workflow:

- Start from messy, mixed-type application data  
- Engineer features that reflect *repayment capacity* and *leverage*  
- Compare multiple models under a **cost-aware** lens (false approvals vs false rejections)  
- Recommend a deployable decision policy (thresholds + human oversight)  

---

## Dataset

A historical loan application dataset containing demographic and financial signals, such as:

- Household info (e.g., number of dependents)
- Education & employment indicators
- Income, requested loan, loan term
- Credit score (CIBIL)
- Asset values (residential / commercial / luxury / bank)

Target label
- `Approved` (1) / `Rejected` (0)

---

## ML pipeline

### 1) Data preparation
- Dropped identifier-style fields (non-predictive)
- Stratified train/test split for stable evaluation
- One-hot encoding for categorical variables
- Standardization for continuous features

### 2) Feature engineering (domain-inspired)
To better represent affordability and risk, I created features such as:

- **Total assets** = sum of multiple asset categories  
- **Loan-to-income ratio** (affordability pressure)
- **Loan-to-total-assets ratio** (leverage)
- **Income per dependent** (household burden)

These features are designed to capture the *financial story* behind each application, not just raw numbers.

---

## Models evaluated

I trained and compared four supervised classifiers:

- **Logistic Regression** (interpretable baseline)
- **Decision Tree** (rule-based nonlinearity)
- **Random Forest** (robust ensemble)
- **Gradient Boosting** (strong sequential ensemble)

To handle class imbalance and reduce risky approvals, I applied **class balancing** for tree-based models.

---

## Evaluation

I evaluated models on a held-out test set using:

- Accuracy
- Precision / Recall / F1
- ROC-AUC
- Confusion matrix analysis (false approvals vs false rejections)

This matters because:
- **False approvals** can directly increase default risk  
- **False rejections** can hurt revenue + user experience  

---

## Key results (summary)

| Model | Accuracy | Precision (Approved) | Recall (Approved) | ROC-AUC |
|------|----------|-----------------------|-------------------|--------|
| Logistic Regression | 0.924 | 0.955 | 0.921 | 0.974 |
| Random Forest | 0.982 | ~0.985 | ~0.987 | 0.999 |

**Takeaway:** Logistic Regression is already strong, but **Random Forest provides near-perfect separation**, minimizing both false approvals and false rejections.

---

## What the model learned (business insights)

Across exploratory analysis + model behavior, the strongest signals align with credit-risk intuition:

- **Credit score (CIBIL)** is highly informative  
- **Income and assets** correlate strongly with approval  
- High **loan-to-income** or **loan-to-assets** ratios increase rejection likelihood  
- Lower **income per dependent** indicates household pressure and higher risk  

---

## Recommended deployment policy (how this becomes a product)

Instead of making every decision fully automatic, the model’s probability score supports a **3-tier workflow**:

- **Approve** if probability ≥ high threshold  
- **Decline** if probability ≤ low threshold  
- **Manual review** for borderline cases  

This keeps decisions scalable **without removing human judgement** where it matters most.

---

## Ethics & governance (important, even for coursework)

Automated lending decisions can be harmful if deployed carelessly. Key considerations include:

- **Bias & fairness:** even without sensitive attributes, proxies can exist  
- **Explainability:** ensembles need clear reason codes (e.g., low score / high leverage)  
- **Privacy & compliance:** strict access control and responsible data handling  
- **Human oversight:** especially for edge cases and limited credit history applicants  

---

## Repository contents

Typical contents in this repo:

- `*.ipynb` — full analysis + modeling pipeline (EDA → features → training → evaluation)
- `INT303report.pdf` — final written report (methods, results, recommendations)

> Note: Raw datasets are often excluded from GitHub uploads. This project supports downloading the dataset programmatically inside the notebook.

---
