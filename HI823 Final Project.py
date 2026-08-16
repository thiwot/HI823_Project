#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from pgmpy.estimators import HillClimbSearch, BIC, MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def e_value(odds_ratio):
    rr = max(odds_ratio, 1 / odds_ratio)
    return rr + np.sqrt(rr * (rr - 1))


# In[2]:


DATA_PATH = "HI823_Data/diabetic_data.csv"
df_raw = pd.read_csv(DATA_PATH, keep_default_na=False, na_values=["?"])
print(df_raw.shape)
df_raw.head()


# In[3]:


df = df_raw.replace("?", np.nan)
df = df.drop(columns=["weight", "payer_code", "medical_specialty"])
df = df.drop_duplicates(subset="patient_nbr", keep="first")
df = df.drop(columns=["encounter_id", "patient_nbr"]).reset_index(drop=True)

age_map = {f"[{i}-{i+10})": i + 5 for i in range(0, 100, 10)}
df["age_num"] = df["age"].map(age_map)
df = df.drop(columns=["age"])

df["target"] = (df["readmitted"] == "<30").astype(int)
df = df.drop(columns=["readmitted"])

df["treated"] = (df["insulin"] != "No").astype(int)
df = df.drop(columns=["insulin"])

print(df.shape)
df.isna().mean().sort_values(ascending=False).head(10)


# In[4]:


def icd9_group(code):
    '''Collapse raw ICD-9 diagnosis codes into broad clinical categories,
    following the grouping convention used in Strack et al. (2014).'''
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith("250"):
        return "Diabetes"
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        val = float(code)
    except ValueError:
        return "Other"
    if 390 <= val <= 459 or val == 785:
        return "Circulatory"
    if 460 <= val <= 519 or val == 786:
        return "Respiratory"
    if 520 <= val <= 579 or val == 787:
        return "Digestive"
    if 800 <= val <= 999:
        return "Injury"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 580 <= val <= 629 or val == 788:
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"

for c in ["diag_1", "diag_2", "diag_3"]:
    df[c] = df[c].apply(icd9_group)

print(df["diag_1"].value_counts())


# In[5]:


cat_cols = df.select_dtypes(include="object").columns.tolist()
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df_enc = df_enc.fillna(df_enc.median(numeric_only=True))

y = df_enc["target"]
treat = df_enc["treated"]
X = df_enc.drop(columns=["target", "treated"])
print(f"{X.shape[1]} candidate predictors, {X.shape[0]} patients")


# In[6]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
y.value_counts(normalize=True).plot(kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"])
axes[0].set_title("Outcome balance: 30-day readmission")
axes[0].set_xticklabels(["No (0)", "Yes (1)"], rotation=0)

treat.value_counts(normalize=True).plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"])
axes[1].set_title("Exposure balance: insulin therapy")
axes[1].set_xticklabels(["Not treated (0)", "Treated (1)"], rotation=0)
plt.tight_layout()

plt.show()

print("Outcome prevalence:", y.mean().round(3))
print("Treatment prevalence:", treat.mean().round(3))


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
sm_smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = sm_smote.fit_resample(X_train, y_train)
print("Before SMOTE:", y_train.value_counts(normalize=True).round(3).to_dict())
print("After SMOTE :", y_train_bal.value_counts(normalize=True).round(3).to_dict())


# In[8]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_cv = LogisticRegressionCV(
    penalty="l1", solver="liblinear", Cs=10, cv=5,
    max_iter=2000, random_state=RANDOM_STATE
)
lasso_cv.fit(np.column_stack([X_scaled, treat]), y)

coef_names = list(X.columns) + ["treated"]
lasso_coefs = pd.Series(lasso_cv.coef_[0], index=coef_names).sort_values(key=abs, ascending=False)
selected = lasso_coefs[lasso_coefs != 0].index.tolist()
confounders = [c for c in selected if c != "treated"]

used_fallback = len(confounders) < 3
if used_fallback:
    ridge = LogisticRegression(penalty="l2", C=1.0, max_iter=2000).fit(X_scaled, y)
    ridge_coefs = pd.Series(np.abs(ridge.coef_[0]), index=X.columns).sort_values(ascending=False)
    confounders = ridge_coefs.head(10).index.tolist()
    print(f"LASSO shrank all coefficients to zero (0 of {X.shape[1]} survived) — "
          f"using ridge-regression fallback. Top confounders by |coefficient|:")
    print(ridge_coefs.head(10))
else:
    print(f"LASSO selected {len(confounders)} of {X.shape[1]} candidate confounders")
    print(lasso_coefs.reindex(confounders).sort_values(key=abs, ascending=False))


# In[9]:


ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X[confounders], treat)
ps = np.clip(ps_model.predict_proba(X[confounders])[:, 1], 0.02, 0.98)

plt.figure(figsize=(6, 4))
sns.kdeplot(ps[treat == 1], label="Treated", fill=True)
sns.kdeplot(ps[treat == 0], label="Control", fill=True)
plt.xlabel("Propensity score"); plt.legend(); plt.title("Propensity score overlap (positivity check)")
plt.show()


# In[10]:


treat_idx = np.where(treat.values == 1)[0]
ctrl_idx = np.where(treat.values == 0)[0]

nn = NearestNeighbors(n_neighbors=1).fit(ps[ctrl_idx].reshape(-1, 1))
_, ind = nn.kneighbors(ps[treat_idx].reshape(-1, 1))
matched_ctrl_idx = ctrl_idx[ind.flatten()]
print(f"Matched {len(treat_idx)} treated patients to {len(set(matched_ctrl_idx))} unique controls")

def smd(a, b):
    '''Standardized mean difference — covariate-balance diagnostic'''
    return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-9)

balance_check = []
for c in confounders[:8]:
    before = smd(X.loc[treat_idx, c], X.loc[ctrl_idx, c])
    after = smd(X.loc[treat_idx, c], X.loc[matched_ctrl_idx, c])
    balance_check.append({"covariate": c, "smd_before": before, "smd_after": after})
balance_df = pd.DataFrame(balance_check)
balance_df


# In[11]:


matched_idx = np.concatenate([treat_idx, matched_ctrl_idx])
matched_treat = treat.iloc[matched_idx]
matched_y = y.iloc[matched_idx]

matched_model = LogisticRegression(max_iter=1000).fit(matched_treat.values.reshape(-1, 1), matched_y)
matched_or = np.exp(matched_model.coef_[0][0])
print("Matched-sample odds ratio (insulin therapy vs none):", round(matched_or, 3))

risk_treated = y.iloc[treat_idx].mean()
risk_matched_ctrl = y.iloc[matched_ctrl_idx].mean()
print("Risk difference (matched):", round(risk_treated - risk_matched_ctrl, 4))


# In[12]:


iptw = treat / ps + (1 - treat) / (1 - ps)
iptw_model = LogisticRegression(max_iter=1000).fit(treat.values.reshape(-1, 1), y, sample_weight=iptw)
iptw_or = np.exp(iptw_model.coef_[0][0])
print("IPTW-adjusted odds ratio (insulin therapy vs none):", round(iptw_or, 3))


# In[13]:


common_min = max(ps[treat.values == 1].min(), ps[treat.values == 0].min())
common_max = min(ps[treat.values == 1].max(), ps[treat.values == 0].max())
in_support = (ps >= common_min) & (ps <= common_max)
print(f"Common support range: [{common_min:.3f}, {common_max:.3f}]")
print(f"Dropping {(~in_support).sum()} of {len(ps)} patients outside common support "
      f"({(~in_support).mean()*100:.1f}%)")

ps_cs = ps[in_support]
treat_cs = treat[in_support].reset_index(drop=True)
y_cs = y[in_support].reset_index(drop=True)

treat_idx_cs = np.where(treat_cs.values == 1)[0]
ctrl_idx_cs = np.where(treat_cs.values == 0)[0]
nn_cs = NearestNeighbors(n_neighbors=1).fit(ps_cs[ctrl_idx_cs].reshape(-1, 1))
_, ind_cs = nn_cs.kneighbors(ps_cs[treat_idx_cs].reshape(-1, 1))
matched_ctrl_idx_cs = ctrl_idx_cs[ind_cs.flatten()]
print(f"[Common support] Matched {len(treat_idx_cs)} treated patients to "
      f"{len(set(matched_ctrl_idx_cs))} unique controls")

matched_idx_cs = np.concatenate([treat_idx_cs, matched_ctrl_idx_cs])
matched_treat_cs = treat_cs.iloc[matched_idx_cs]
matched_y_cs = y_cs.iloc[matched_idx_cs]
matched_model_cs = LogisticRegression(max_iter=1000).fit(matched_treat_cs.values.reshape(-1, 1), matched_y_cs)
matched_or_cs = np.exp(matched_model_cs.coef_[0][0])
print("[Common support] Matched-sample odds ratio:", round(matched_or_cs, 3))

risk_treated_cs = y_cs.iloc[treat_idx_cs].mean()
risk_matched_ctrl_cs = y_cs.iloc[matched_ctrl_idx_cs].mean()
print("[Common support] Risk difference (matched):", round(risk_treated_cs - risk_matched_ctrl_cs, 4))

iptw_cs = treat_cs / ps_cs + (1 - treat_cs) / (1 - ps_cs)
iptw_model_cs = LogisticRegression(max_iter=1000).fit(treat_cs.values.reshape(-1, 1), y_cs, sample_weight=iptw_cs)
iptw_or_cs = np.exp(iptw_model_cs.coef_[0][0])
print("[Common support] IPTW-adjusted odds ratio:", round(iptw_or_cs, 3))

print("[Common support] E-value (matched):", round(e_value(matched_or_cs), 3))
print("[Common support] E-value (IPTW):   ", round(e_value(iptw_or_cs), 3))


# In[14]:


mediator = "num_medications"
med_binary = (X[mediator] > X[mediator].median()).astype(int)

a_model = LogisticRegression(max_iter=1000).fit(treat.values.reshape(-1, 1), med_binary)
a_coef = a_model.coef_[0][0]

b_model = LogisticRegression(max_iter=1000).fit(np.column_stack([treat, X[mediator]]), y)
b_coef = b_model.coef_[0][1]
direct_coef = b_model.coef_[0][0]
indirect_effect = a_coef * b_coef

print(f"Path a (treatment -> mediator):   {a_coef:.4f}")
print(f"Path b (mediator -> outcome):     {b_coef:.4f}")
print(f"Direct effect (treatment -> outcome, adjusted for mediator): {direct_coef:.4f}")
print(f"Approx. indirect (mediated) effect: {indirect_effect:.4f}")


# In[15]:


bn_candidates = [c for c in confounders if X[c].nunique() <= 10][:6]
bn_df = pd.concat([X[bn_candidates], treat.rename("treated"), y.rename("target")], axis=1).dropna()

cont_vars = [c for c in bn_candidates if bn_df[c].nunique() > 5]
if cont_vars:
    disc = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
    bn_df[cont_vars] = disc.fit_transform(bn_df[cont_vars]).astype(int)
bn_df = bn_df.astype(str)

hc = HillClimbSearch(bn_df)
best_structure = hc.estimate(scoring_method=BIC(bn_df), max_iter=50)
edges = list(best_structure.edges())

if len(edges) == 0:
    edges = list(dict.fromkeys([(c, "target") for c in bn_candidates[:4]] + [("treated", "target")]))
    print("Score-based search found no edges beating the BIC penalty; using an "
          "expert-specified fallback structure instead:", edges)
else:
    print("Learned network edges:", edges)

bn_model = DiscreteBayesianNetwork(edges)
mle = MaximumLikelihoodEstimator(bn_model, bn_df)
bn_model.add_cpds(*mle.get_parameters())
assert bn_model.check_model()

markov_blanket = bn_model.get_markov_blanket("target")
print("\nDirect predictors of readmission (Markov blanket of 'target'):", markov_blanket)


# In[16]:


infer = VariableElimination(bn_model)

def predict_case(input_dict):
    '''Return predicted readmission probability distribution for a patient profile'''
    evidence = {k: str(v) for k, v in input_dict.items() if k in bn_model.nodes() and k != "target"}
    result = infer.query(variables=["target"], evidence=evidence, show_progress=False)
    return dict(zip(result.state_names["target"], result.values))

example_patient = {"treated": "1"}
print("predict_case example:", predict_case(example_patient))


# In[17]:


# Figure 1: causal network diagram (DAG)
markov_blanket_set = set(markov_blanket)
target_node = "target"

G = nx.DiGraph()
G.add_edges_from(edges)
pos = nx.spring_layout(G, seed=42, k=1.2)

node_colors = []
for n in G.nodes():
    if n == target_node:
        node_colors.append("#C44E52")
    elif n in markov_blanket_set:
        node_colors.append("#55A868")
    elif n == "treated":
        node_colors.append("#4C72B0")
    else:
        node_colors.append("#DDDDDD")

plt.figure(figsize=(9, 7))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200, edgecolors="black", linewidths=1)
nx.draw_networkx_labels(G, pos, font_size=8)
nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=18, edge_color="#555555", connectionstyle="arc3,rad=0.08")
plt.title("Learned Causal Network Structure\n(green = Markov blanket; blue = treatment; red = outcome)", fontsize=11)
plt.axis("off")
plt.tight_layout()
plt.show()


# In[18]:


log_reg = LogisticRegression(max_iter=2000).fit(X_train_bal, y_train_bal)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE).fit(X_train_bal, y_train_bal)
xgb = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    eval_metric="logloss", random_state=RANDOM_STATE
).fit(X_train_bal, y_train_bal)

for name, model in [("Logistic Regression", log_reg), ("Random Forest", rf), ("XGBoost", xgb)]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    print(f"{name}: AUC={roc_auc_score(y_test, proba):.3f}  "
          f"Precision={precision_score(y_test, pred):.3f}  "
          f"Recall={recall_score(y_test, pred):.3f}")


# In[19]:


print("E-value for the matched-sample odds ratio:", round(e_value(matched_or), 3))
print("E-value for the IPTW-adjusted odds ratio:  ", round(e_value(iptw_or), 3))


# In[20]:


confounder_idx = [X.columns.get_loc(c) for c in confounders]
Xc_scaled = X_scaled[:, confounder_idx]

full_model = LogisticRegression(max_iter=1000).fit(np.column_stack([Xc_scaled, treat]), y)
p_full = full_model.predict_proba(np.column_stack([Xc_scaled, treat]))[:, 1]
ll_full = -log_loss(y, p_full, normalize=False)

p_null = y.mean()
ll_null = np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null))
mcfadden_r2 = 1 - ll_full / ll_null
print("McFadden's pseudo R^2:", round(mcfadden_r2, 4))


# In[21]:


readmit_by_exposure = pd.crosstab(
    df["treated"], df_raw.loc[df.index, "readmitted"], normalize="index"
)[["NO", ">30", "<30"]]
print(readmit_by_exposure.round(3))

readmit_by_exposure.plot(kind="bar", stacked=True, figsize=(6, 4), color=["#4C72B0", "#DD8452", "#C44E52"])
plt.ylabel("Proportion of patients")
plt.title("Readmission timing by insulin exposure group")
plt.legend(title="Readmitted", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks([0, 1], ["Not treated", "Treated"], rotation=0)
plt.tight_layout()
plt.show()

print("\nRaw counts:")
print(pd.crosstab(df["treated"], df_raw.loc[df.index, "readmitted"]))


# In[22]:


raw2 = df_raw.replace("?", np.nan)
raw2 = raw2.drop(columns=["weight", "payer_code", "medical_specialty"])
raw2 = raw2.drop_duplicates(subset="patient_nbr", keep="first")
raw2 = raw2.drop(columns=["encounter_id", "patient_nbr"]).reset_index(drop=True)
insulin_raw = raw2["insulin"]
assert len(insulin_raw) == len(X), "Row count mismatch"

mask = insulin_raw.isin(["Up", "Steady", "No"])
treat2 = (insulin_raw[mask] == "Up").astype(int).reset_index(drop=True)
X2 = X[mask].reset_index(drop=True)
y2 = y[mask].reset_index(drop=True)
print(f"Excluded {(~mask).sum()} patients whose insulin dose was decreased ('Down')")
print("Exposure prevalence (dose increased vs not):")
print(treat2.value_counts(normalize=True).round(3))

scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)
lasso2 = LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=8, cv=5, max_iter=2000, random_state=RANDOM_STATE)
lasso2.fit(np.column_stack([X2_scaled, treat2]), y2)
coefs2 = pd.Series(lasso2.coef_[0], index=list(X2.columns) + ["treated"])
confounders2 = [c for c in coefs2[coefs2 != 0].index.tolist() if c != "treated"]
if len(confounders2) < 3:
    ridge2 = LogisticRegression(penalty="l2", C=1.0, max_iter=2000).fit(X2_scaled, y2)
    confounders2 = pd.Series(np.abs(ridge2.coef_[0]), index=X2.columns).sort_values(ascending=False).head(10).index.tolist()
print(f"Confounders: {confounders2}")

ps_model2 = LogisticRegression(max_iter=1000).fit(X2[confounders2], treat2)
ps2 = np.clip(ps_model2.predict_proba(X2[confounders2])[:, 1], 0.02, 0.98)

plt.figure(figsize=(6, 4))
sns.kdeplot(ps2[treat2 == 1], label="Dose increased", fill=True)
sns.kdeplot(ps2[treat2 == 0], label="No change", fill=True)
plt.xlabel("Propensity score"); plt.legend(); plt.title("Propensity overlap: dose-increase exposure")
plt.show()

treat_idx2 = np.where(treat2.values == 1)[0]
ctrl_idx2 = np.where(treat2.values == 0)[0]
nn2 = NearestNeighbors(n_neighbors=1).fit(ps2[ctrl_idx2].reshape(-1, 1))
_, ind2 = nn2.kneighbors(ps2[treat_idx2].reshape(-1, 1))
matched_ctrl_idx2 = ctrl_idx2[ind2.flatten()]
print(f"Matched {len(treat_idx2)} to {len(set(matched_ctrl_idx2))} unique controls")

matched_treat2 = treat2.iloc[np.concatenate([treat_idx2, matched_ctrl_idx2])]
matched_y2 = y2.iloc[np.concatenate([treat_idx2, matched_ctrl_idx2])]
matched_or2 = np.exp(LogisticRegression(max_iter=1000).fit(matched_treat2.values.reshape(-1,1), matched_y2).coef_[0][0])
print("Matched OR (dose increase vs no change):", round(matched_or2, 3))

iptw2 = treat2 / ps2 + (1 - treat2) / (1 - ps2)
iptw_or2 = np.exp(LogisticRegression(max_iter=1000).fit(treat2.values.reshape(-1,1), y2, sample_weight=iptw2).coef_[0][0])
print("IPTW OR (dose increase vs no change):", round(iptw_or2, 3))
print("E-value (matched):", round(e_value(matched_or2), 3))
print("E-value (IPTW):", round(e_value(iptw_or2), 3))


# In[23]:


raw3 = df_raw.replace("?", np.nan)
raw3 = raw3.drop(columns=["weight", "payer_code", "medical_specialty"])
raw3 = raw3.drop_duplicates(subset="patient_nbr", keep="first")
raw3 = raw3.drop(columns=["encounter_id", "patient_nbr"]).reset_index(drop=True)
change_raw = raw3["change"]
assert len(change_raw) == len(X), "Row count mismatch"

treat3 = (change_raw == "Ch").astype(int).reset_index(drop=True)
print("Exposure prevalence (any med change vs none):")
print(treat3.value_counts(normalize=True).round(3))

scaler3 = StandardScaler()
X3_scaled = scaler3.fit_transform(X)
lasso3 = LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=8, cv=5, max_iter=2000, random_state=RANDOM_STATE)
lasso3.fit(np.column_stack([X3_scaled, treat3]), y)
coefs3 = pd.Series(lasso3.coef_[0], index=list(X.columns) + ["treated"])
confounders3 = [c for c in coefs3[coefs3 != 0].index.tolist() if c != "treated"]
if len(confounders3) < 3:
    ridge3 = LogisticRegression(penalty="l2", C=1.0, max_iter=2000).fit(X3_scaled, y)
    confounders3 = pd.Series(np.abs(ridge3.coef_[0]), index=X.columns).sort_values(ascending=False).head(10).index.tolist()
print(f"Confounders: {confounders3}")

ps_model3 = LogisticRegression(max_iter=1000).fit(X[confounders3], treat3)
ps3 = np.clip(ps_model3.predict_proba(X[confounders3])[:, 1], 0.02, 0.98)

treat_idx3 = np.where(treat3.values == 1)[0]
ctrl_idx3 = np.where(treat3.values == 0)[0]
nn3 = NearestNeighbors(n_neighbors=1).fit(ps3[ctrl_idx3].reshape(-1, 1))
_, ind3 = nn3.kneighbors(ps3[treat_idx3].reshape(-1, 1))
matched_ctrl_idx3 = ctrl_idx3[ind3.flatten()]
print(f"Matched {len(treat_idx3)} to {len(set(matched_ctrl_idx3))} unique controls")

matched_treat3 = treat3.iloc[np.concatenate([treat_idx3, matched_ctrl_idx3])]
matched_y3 = y.iloc[np.concatenate([treat_idx3, matched_ctrl_idx3])]
matched_or3 = np.exp(LogisticRegression(max_iter=1000).fit(matched_treat3.values.reshape(-1,1), matched_y3).coef_[0][0])
print("Matched OR (med change vs none):", round(matched_or3, 3))

iptw3 = treat3 / ps3 + (1 - treat3) / (1 - ps3)
iptw_or3 = np.exp(LogisticRegression(max_iter=1000).fit(treat3.values.reshape(-1,1), y, sample_weight=iptw3).coef_[0][0])
print("IPTW OR (med change vs none):", round(iptw_or3, 3))
print("E-value (matched):", round(e_value(matched_or3), 3))
print("E-value (IPTW):", round(e_value(iptw_or3), 3))


# In[24]:


severity_cols = [c for c in X.columns if c.startswith("A1Cresult_") or c.startswith("max_glu_serum_")]
print("Severity-proxy columns found:", severity_cols)

confounders_severity = list(dict.fromkeys(confounders + severity_cols))
print(f"Confounder set expanded from {len(confounders)} to {len(confounders_severity)} variables")

ps_model_sev = LogisticRegression(max_iter=1000).fit(X[confounders_severity], treat)
ps_sev = np.clip(ps_model_sev.predict_proba(X[confounders_severity])[:, 1], 0.02, 0.98)

treat_idx_sev = np.where(treat.values == 1)[0]
ctrl_idx_sev = np.where(treat.values == 0)[0]
nn_sev = NearestNeighbors(n_neighbors=1).fit(ps_sev[ctrl_idx_sev].reshape(-1, 1))
_, ind_sev = nn_sev.kneighbors(ps_sev[treat_idx_sev].reshape(-1, 1))
matched_ctrl_idx_sev = ctrl_idx_sev[ind_sev.flatten()]
print(f"Matched {len(treat_idx_sev)} to {len(set(matched_ctrl_idx_sev))} unique controls")

matched_treat_sev = treat.iloc[np.concatenate([treat_idx_sev, matched_ctrl_idx_sev])]
matched_y_sev = y.iloc[np.concatenate([treat_idx_sev, matched_ctrl_idx_sev])]
matched_or_sev = np.exp(LogisticRegression(max_iter=1000).fit(matched_treat_sev.values.reshape(-1,1), matched_y_sev).coef_[0][0])
print("Matched OR, with severity proxies (any insulin vs none):", round(matched_or_sev, 3))

iptw_sev = treat / ps_sev + (1 - treat) / (1 - ps_sev)
iptw_or_sev = np.exp(LogisticRegression(max_iter=1000).fit(treat.values.reshape(-1,1), y, sample_weight=iptw_sev).coef_[0][0])
print("IPTW OR, with severity proxies (any insulin vs none):", round(iptw_or_sev, 3))
print("E-value (matched):", round(e_value(matched_or_sev), 3))
print("E-value (IPTW):", round(e_value(iptw_or_sev), 3))

print("\nFor comparison, primary (non-severity-adjusted) estimates were:")
print(f"  Matched OR: {matched_or:.3f} | IPTW OR: {iptw_or:.3f}")


# In[25]:


confounders_sg = ["number_inpatient", "discharge_disposition_id", "age_num", "diabetesMed_Yes", "time_in_hospital"]
confounders_sg = [c for c in confounders_sg if c in X.columns]

elderly = (X["age_num"] >= 65).astype(int)
high_severity = (X["number_inpatient"] > 0).astype(int)

def interaction_test(subgroup, label):
    Xd = X[confounders_sg].copy()
    Xd["treated"] = treat.values
    Xd["subgroup"] = subgroup.values
    Xd["treated_x_subgroup"] = Xd["treated"] * Xd["subgroup"]
    Xd = sm.add_constant(Xd.astype(float))
    model = sm.Logit(y.values, Xd).fit(disp=0)
    coef = model.params["treated_x_subgroup"]
    pval = model.pvalues["treated_x_subgroup"]
    print(f"\n--- Effect modification by {label} ---")
    print(f"Interaction coefficient (treated x {label}): {coef:.4f}, p={pval:.4f}")
    print(f"Odds ratio of the interaction term: {np.exp(coef):.3f}")
    for grp_val, grp_name in [(0, f"Not {label}"), (1, label)]:
        idx = subgroup == grp_val
        or_sub = np.exp(LogisticRegression(max_iter=1000).fit(treat[idx].values.reshape(-1, 1), y[idx]).coef_[0][0])
        print(f"  Within-subgroup OR ({grp_name}, n={idx.sum()}): {or_sub:.3f}")

interaction_test(elderly, "elderly (65+)")
interaction_test(high_severity, "prior_inpatient_history")


# In[ ]:




