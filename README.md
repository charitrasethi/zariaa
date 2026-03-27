# Zaria Fashion Analytics Dashboard

A data-driven analytics platform for Zaria Fashion — a pan-India ethnic wear brand.

## Features
- **Descriptive Analysis** — Demographics, psychographics, product ownership
- **Diagnostic Analysis** — Correlations, cross-tabs, causal patterns
- **Customer Clustering** — K-Means segmentation with business personas
- **Association Rule Mining** — Apriori with support, confidence & lift
- **Classification** — Random Forest, XGBoost, Logistic Regression (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Regression & CLV** — Annual spend prediction & customer lifetime value tiering
- **New Customer Predictor** — Upload CSV → instant predictions + marketing playbook

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub → Streamlit Deployment

1. Push all files to a GitHub repository (no sub-folders needed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Deploy!

## Dataset
`zaria_25col_survey.csv` — 1,200 synthetic survey respondents, 25 columns, pan-India distribution with realistic correlations, noise & outliers.
