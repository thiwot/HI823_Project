# HI823 Project: Insulin Therapy and 30-Day Readmission: A Causal Inference Analysis

Causal analysis project for HI823, Summer 2026. Estimates whether inpatient
insulin therapy affects 30-day hospital readmission among diabetic patients, using
propensity score matching, IPTW, a Bayesian network, mediation analysis, and E-value
sensitivity analysis.

**Main finding:** the estimated effect is small and unstable in direction across
estimators and specifications — see the full report for details.

## Repo contents

- `HI823 Final Project.py` — full, executable analysis notebook
- `requirements.txt` — Python dependencies
- `README.md` — instructions for downloading the dataset (not included here — see below)

## Getting the data

This project uses the **Diabetes 130-US Hospitals** dataset (Strack et al., 2014),
101,766 encounters across 130 US hospitals.

- Kaggle: https://www.kaggle.com/datasets/brandao/diabetes
- Or UCI: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Download `diabetic_data.csv` and place it in a `data/` folder before running the notebook.

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook diabetes_causal_analysis.ipynb
```
Run all cells top to bottom (Kernel → Restart & Run All).

## Methods covered

- Data balancing (SMOTE)
- LASSO-based confounder selection
- Propensity score matching and IPTW
- Common-support trimming and alternative exposure definitions (robustness checks)
- Bayesian network / Markov blanket analysis
- Mediation analysis
- E-value sensitivity analysis
- Predictive model comparison (logistic regression, random forest, XGBoost)
