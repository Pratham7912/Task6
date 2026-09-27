# Customer Churn Analysis & Retention Strategy — Integrative Capstone Project

An end-to-end data science pipeline for a telecommunications company: **who is going to churn, what kinds of customers do we have, and what should we do about it?** Combines unsupervised customer segmentation with supervised churn prediction on the real-world IBM Telco Customer Churn dataset.

## 📊 Overview

This capstone integrates the full data science lifecycle — acquisition, cleaning, EDA, feature engineering, unsupervised learning, and supervised learning — into one coherent business analysis, rather than treating each technique in isolation.

**Key results:**
- **Unsupervised**: K-Means identifies 4 behaviorally distinct customer segments (no churn label used) with actual churn rates ranging from **13.7% to 46.6%**
- **Supervised**: Tuned Gradient Boosting classifier predicts churn with **84.4% ROC-AUC** on held-out test data
- **Synthesis**: The two independently-derived models converge on the same churn drivers (contract type, tenure, service engagement) — cross-validating each other
- Overall churn rate: 26.5% | Month-to-month contracts churn at **42.7%** vs. **2.8%** for two-year contracts

## 🗂️ Repository Contents

| File | Description |
|---|---|
| `Week6_Capstone.py` | Full pipeline: data acquisition → cleaning → EDA → feature engineering → K-Means segmentation → churn classification → combined evaluation |
| `Week6_Capstone_Report.docx` | Complete written report: problem statement, methodology, all findings, business recommendations, and reflective discussion |
| `telco_customer_churn.csv` | The dataset used (IBM's public Telco Customer Churn sample data) |
| `figs/` *(generated on run)* | All 13 output visualizations |

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (`KMeans`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `GridSearchCV`, `StratifiedKFold`)
- matplotlib, seaborn

## 🚀 Getting Started

```bash
# clone the repo
git clone https://github.com/<your-username>/telco-churn-capstone.git
cd telco-churn-capstone

# install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# run the analysis
python Week6_Capstone.py
```

All figures are saved to `figs/`, and a full metrics/insights summary is saved to `summary.json`.

## 🔍 Methodology

1. **Data acquisition** — IBM's public Telco Customer Churn dataset (7,043 customers, 21 attributes), loaded directly from source
2. **Data cleaning** — fixed a mis-typed `TotalCharges` column with hidden blank values (correctly imputed as 0 for new customers, not a statistical guess); collapsed redundant categorical levels
3. **EDA** — churn-rate breakdowns by contract type, internet service, payment method, tenure
4. **Feature engineering** — tenure buckets, add-on service count, spend-intensity ratio
5. **Unsupervised learning** — K-Means segmentation (elbow + silhouette to select k=4), fit on behavioral features only, **churn label excluded**
6. **Supervised learning** — 4 classifiers compared via 5-fold CV, hyperparameter-tuned via `GridSearchCV`, final model evaluated on a held-out test set (confusion matrix, ROC/PR curves, feature importance)
7. **Synthesis** — segment-level actual churn rate vs. model-predicted average risk, showing the two approaches independently agree

## 📈 Sample Output

- Customer segments visualized in PCA space, profiled by tenure/spend/engagement
- ROC and precision-recall curves for the final churn model
- Feature importance ranking (contract type, tenure, and internet service dominate)
- Combined segment-risk chart validating unsupervised and supervised findings against each other

## 💡 Business Recommendations

- Prioritize the highest-risk segment (premium-priced, under-served, 46.6% churn) with targeted add-on bundling offers
- Incentivize upgrades from month-to-month to longer contracts — the single highest-leverage retention lever identified
- Investigate fiber-optic service quality/pricing, given its disproportionately high churn rate
- Deploy the model as a monthly scored risk list for a retention team's outreach queue

## 📄 Data Source & License

Dataset: [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv), a public sample dataset distributed by IBM for its Watson Studio / Cognos Analytics tutorials. Code is provided for educational purposes.
