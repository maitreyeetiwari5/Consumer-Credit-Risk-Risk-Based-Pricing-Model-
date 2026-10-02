"""
Consumer Credit Risk & Risk-Based Pricing Model
================================================
End-to-end pipeline on the public LendingClub (2007-2010) loan data:

  1. Probability-of-default (PD) modeling: logistic regression vs gradient boosting
  2. Validation: hold-out AUC / Gini / KS, calibration, bootstrap confidence intervals
  3. Hypothesis tests on the drivers of default
  4. Risk-based pricing: break-even APR for every loan given its PD
  5. Approval-cutoff optimisation (approval rate vs loss rate vs profit) + LGD sensitivity
  6. Model monitoring: PSI on score and features, plus a SIMULATED drift stress test

Run:  python src/credit_risk.py
All outputs are written to ./outputs (CSVs, PNGs, results.json).
Fixed random seed => results are reproducible.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import brentq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "loan_data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

DATA_URL = ("https://raw.githubusercontent.com/zaid7860/LendingClub.com-Fraud-analysis/"
            "master/loan_data.csv")  # public mirror of the cleaned LendingClub 2007-2010 file


def ensure_data():
    if not DATA.exists():
        import urllib.request
        DATA.parent.mkdir(exist_ok=True)
        print(f"Downloading dataset to {DATA} ...")
        urllib.request.urlretrieve(DATA_URL, DATA)


# ---- Business assumptions (NOT estimated from the data - see README) ----------
TERM_MONTHS = 36          # dataset has no term column; 36 months assumed
BASE_LGD = 0.50           # loss as share of original principal, net of payments before default
LGD_GRID = [0.30, 0.40, 0.50, 0.60, 0.70]
COST_OF_FUNDS = 0.04      # funding + servicing cost, % of average outstanding balance per year
MAX_APR = 0.36            # cap when solving for break-even rate
N_BOOT = 1000

rng = np.random.default_rng(SEED)
results = {}


# =============================================================================
# 1. Data & features
# =============================================================================
def load_and_engineer(path):
    df = pd.read_csv(path)
    df.columns = [c.replace(".", "_") for c in df.columns]
    df = df.rename(columns={"not_fully_paid": "default"})

    # Principal implied by the installment, rate and (assumed) term.
    i = df["int_rate"] / 12
    df["principal"] = df["installment"] * (1 - (1 + i) ** -TERM_MONTHS) / i

    feats = pd.DataFrame(index=df.index)
    feats["fico"] = df["fico"]
    feats["dti"] = df["dti"]
    feats["log_income"] = df["log_annual_inc"]
    feats["credit_history_yrs"] = df["days_with_cr_line"] / 365.25
    feats["log_revol_bal"] = np.log1p(df["revol_bal"])
    feats["revol_util"] = df["revol_util"]
    feats["inq_6m"] = df["inq_last_6mths"]
    feats["delinq_2y"] = df["delinq_2yrs"]
    feats["pub_rec"] = df["pub_rec"]
    feats["payment_to_income"] = df["installment"] * 12 / np.exp(df["log_annual_inc"])
    feats["purpose"] = df["purpose"]
    return df, feats


NUM = ["fico", "dti", "log_income", "credit_history_yrs", "log_revol_bal",
       "revol_util", "inq_6m", "delinq_2y", "pub_rec", "payment_to_income"]
CAT = ["purpose"]

# =============================================================================
# 2. Metrics
# =============================================================================
def ks_stat(y, p):
    return stats.ks_2samp(p[y == 1], p[y == 0]).statistic


def psi(expected, actual, bins=10):
    """Population Stability Index using reference-population quantile bins."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def monthly_payment(principal, annual_rate, n=TERM_MONTHS):
    i = np.asarray(annual_rate, dtype=float) / 12
    safe_i = np.where(i == 0, 1e-12, i)
    pmt = principal * safe_i / (1 - (1 + safe_i) ** -n)
    return float(pmt) if np.ndim(pmt) == 0 else pmt


def funding_cost(principal, annual_rate, n=TERM_MONTHS):
    """COST_OF_FUNDS x average outstanding balance (in years) over an amortising schedule."""
    principal = np.asarray(principal, dtype=float)
    i = np.asarray(annual_rate, dtype=float) / 12
    pmt = monthly_payment(principal, annual_rate, n)
    k = np.arange(n).reshape(-1, *([1] * np.ndim(i)))
    balance = principal * (1 + i) ** k - pmt * ((1 + i) ** k - 1) / np.where(i == 0, 1e-12, i)
    cost = COST_OF_FUNDS * balance.sum(axis=0) / 12
    return float(cost) if np.ndim(cost) == 0 else cost


def break_even_rate(pd_, principal, lgd):
    """APR at which expected profit = 0:
       (1-PD)*interest - PD*LGD*P - funding cost = 0."""
    def f(i):
        r = i * 12
        interest = monthly_payment(principal, r) * TERM_MONTHS - principal
        return (1 - pd_) * interest - pd_ * lgd * principal - funding_cost(principal, r)
    hi = MAX_APR / 12
    if f(hi) < 0:
        return np.nan  # no viable price below the APR cap
    if f(1e-9) > 0:
        return 0.0
    return brentq(f, 1e-9, hi) * 12


def expected_profit(pd_, principal, rate, lgd):
    interest = monthly_payment(principal, rate) * TERM_MONTHS - principal
    return (1 - pd_) * interest - pd_ * lgd * principal - funding_cost(principal, rate)


def realized_profit(default, principal, rate, lgd):
    interest = monthly_payment(principal, rate) * TERM_MONTHS - principal
    gross = np.where(default == 1, -lgd * principal, interest)
    return gross - funding_cost(principal, rate)


# =============================================================================
# main
# =============================================================================
ensure_data()
df, X = load_and_engineer(DATA)
y = df["default"].values
results["data"] = {
    "n_loans": int(len(df)),
    "default_rate": round(float(y.mean()), 4),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
}

X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
    X, y, df, test_size=0.30, stratify=y, random_state=SEED
)
results["split"] = {"train": int(len(X_tr)), "test": int(len(X_te)),
                    "train_default_rate": round(float(y_tr.mean()), 4),
                    "test_default_rate": round(float(y_te.mean()), 4)}

# ---------------------------------------------------------------- EDA tables --
fico_bins = pd.cut(df["fico"], [0, 659, 699, 739, 900],
                   labels=["<660", "660-699", "700-739", "740+"])
eda = (df.assign(fico_band=fico_bins).groupby("fico_band", observed=True)["default"]
         .agg(loans="size", default_rate="mean").round(4))
eda.to_csv(OUT / "default_rate_by_fico_band.csv")
chi2, p_chi, dof, _ = stats.chi2_contingency(pd.crosstab(fico_bins, df["default"]))
results["hypothesis_tests"] = {
    "H1_default_rate_differs_by_fico_band": {
        "test": "chi-square", "chi2": round(float(chi2), 1), "dof": int(dof),
        "p_value": float(p_chi),
        "default_rates": eda["default_rate"].to_dict(),
    }
}

# --------------------------------------------------------------- Modelling ----
pre_lr = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
])
pre_gb = ColumnTransformer([
    ("num", "passthrough", NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
])
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

lr_search = GridSearchCV(
    Pipeline([("pre", pre_lr), ("m", LogisticRegression(max_iter=2000))]),
    {"m__C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=cv).fit(X_tr, y_tr)

gb_search = GridSearchCV(
    Pipeline([("pre", pre_gb),
              ("m", HistGradientBoostingClassifier(random_state=SEED, early_stopping=False))]),
    {"m__max_depth": [2, 3], "m__learning_rate": [0.03, 0.06],
     "m__max_iter": [150, 300], "m__l2_regularization": [1.0]},
    scoring="roc_auc", cv=cv).fit(X_tr, y_tr)

lr, gb = lr_search.best_estimator_, gb_search.best_estimator_
p_lr, p_gb = lr.predict_proba(X_te)[:, 1], gb.predict_proba(X_te)[:, 1]
p_tr_lr = lr.predict_proba(X_tr)[:, 1]


def model_row(name, p_te, p_tr, cv_auc, params):
    return {
        "model": name,
        "cv_auc_train": round(float(cv_auc), 4),
        "train_auc": round(roc_auc_score(y_tr, p_tr), 4),
        "test_auc": round(roc_auc_score(y_te, p_te), 4),
        "test_gini": round(2 * roc_auc_score(y_te, p_te) - 1, 4),
        "test_ks": round(ks_stat(y_te, p_te), 4),
        "test_brier": round(brier_score_loss(y_te, p_te), 4),
        "best_params": str(params),
    }


# baseline: FICO-only logistic (what a naive score would give)
fico_only = LogisticRegression().fit(X_tr[["fico"]], y_tr)
p_fico = fico_only.predict_proba(X_te[["fico"]])[:, 1]

perf = pd.DataFrame([
    model_row("Logistic regression", p_lr, p_tr_lr, lr_search.best_score_, lr_search.best_params_),
    model_row("Gradient boosting", p_gb, gb.predict_proba(X_tr)[:, 1], gb_search.best_score_, gb_search.best_params_),
    {"model": "Baseline: FICO only", "cv_auc_train": np.nan,
     "train_auc": round(roc_auc_score(y_tr, fico_only.predict_proba(X_tr[["fico"]])[:, 1]), 4),
     "test_auc": round(roc_auc_score(y_te, p_fico), 4),
     "test_gini": round(2 * roc_auc_score(y_te, p_fico) - 1, 4),
     "test_ks": round(ks_stat(y_te, p_fico), 4),
     "test_brier": round(brier_score_loss(y_te, p_fico), 4), "best_params": ""},
])
perf.to_csv(OUT / "model_performance.csv", index=False)
results["model_performance"] = perf.drop(columns="best_params").to_dict("records")

# ---- Bootstrap CIs on test AUC and the logistic-vs-boosting difference ---------
boot = {"lr": [], "gb": [], "diff": []}
idx_all = np.arange(len(y_te))
for _ in range(N_BOOT):
    b = rng.choice(idx_all, len(idx_all), replace=True)
    if y_te[b].min() == y_te[b].max():
        continue
    a1, a2 = roc_auc_score(y_te[b], p_lr[b]), roc_auc_score(y_te[b], p_gb[b])
    boot["lr"].append(a1); boot["gb"].append(a2); boot["diff"].append(a2 - a1)
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
results["bootstrap_auc_95ci"] = {
    "logistic": ci(boot["lr"]), "gradient_boosting": ci(boot["gb"]),
    "gb_minus_lr": ci(boot["diff"]),
    "diff_ci_includes_zero": bool(np.percentile(boot["diff"], 2.5) <= 0 <= np.percentile(boot["diff"], 97.5)),
}

# ---- Champion selection: prefer the interpretable model unless boosting clearly wins
champion_is_lr = results["bootstrap_auc_95ci"]["diff_ci_includes_zero"] or \
    perf.loc[0, "test_auc"] >= perf.loc[1, "test_auc"]
champ_name = "Logistic regression" if champion_is_lr else "Gradient boosting"
p_champ = p_lr if champion_is_lr else p_gb
results["champion"] = champ_name

# ---- Interpretability: statsmodels Logit on standardised features (p-values) ----
Xs = X_tr[NUM].copy()
mu, sd = Xs.mean(), Xs.std()
Xs = sm.add_constant((Xs - mu) / sd)
logit = sm.Logit(y_tr, Xs).fit(disp=0)
coef = pd.DataFrame({
    "feature": logit.params.index,
    "coef_per_1sd": logit.params.values.round(4),
    "odds_ratio_per_1sd": np.exp(logit.params.values).round(3),
    "p_value": logit.pvalues.values,
}).query("feature != 'const'").sort_values("p_value")
coef.to_csv(OUT / "logit_coefficients.csv", index=False)
results["hypothesis_tests"]["H2_logit_coefficients"] = {
    r.feature: {"odds_ratio_per_1sd": r.odds_ratio_per_1sd, "p_value": float(r.p_value)}
    for r in coef.itertuples()
}

# ---- Decile calibration table (champion) -------------------------------------
cal = pd.DataFrame({"pd": p_champ, "y": y_te})
cal["decile"] = pd.qcut(cal["pd"], 10, labels=False, duplicates="drop") + 1
cal_tbl = cal.groupby("decile").agg(loans=("y", "size"), avg_pred_pd=("pd", "mean"),
                                   actual_default=("y", "mean")).round(4)
cal_tbl.to_csv(OUT / "calibration_by_decile.csv")
top = cal_tbl.iloc[-1]["actual_default"]; bot = cal_tbl.iloc[0]["actual_default"]
results["calibration"] = {
    "top_decile_actual_default": float(top), "bottom_decile_actual_default": float(bot),
    "top_to_bottom_ratio": round(float(top / bot), 1),
    "mean_pred_pd": round(float(p_champ.mean()), 4), "actual_default_rate": round(float(y_te.mean()), 4),
}

# =============================================================================
# 4. Risk-based pricing (test set)
# =============================================================================
te = df_te.copy().reset_index(drop=True)
te["pd"] = p_champ
te["break_even_apr"] = [break_even_rate(p, P, BASE_LGD) for p, P in zip(te["pd"], te["principal"])]
te["actual_apr"] = te["int_rate"]
te["margin_vs_break_even"] = te["actual_apr"] - te["break_even_apr"]
te["exp_profit"] = [expected_profit(p, P, r, BASE_LGD) for p, P, r in zip(te["pd"], te["principal"], te["actual_apr"])]
te["real_profit"] = realized_profit(te["default"].values, te["principal"].values, te["actual_apr"].values, BASE_LGD)
te["pd_quintile"] = pd.qcut(te["pd"], 5, labels=["Q1 (lowest risk)", "Q2", "Q3", "Q4", "Q5 (highest risk)"])

price_tbl = te.groupby("pd_quintile", observed=True).agg(
    loans=("pd", "size"), avg_pd=("pd", "mean"), actual_default=("default", "mean"),
    avg_charged_apr=("actual_apr", "mean"), avg_break_even_apr=("break_even_apr", "mean"),
    pct_underpriced=("margin_vs_break_even", lambda s: (s < 0).mean()),
    avg_realized_profit_per_loan=("real_profit", "mean")).round(4)
price_tbl.to_csv(OUT / "pricing_by_risk_quintile.csv")
results["pricing"] = {
    "assumptions": {"term_months": TERM_MONTHS, "lgd": BASE_LGD, "cost_of_funds": COST_OF_FUNDS, "apr_cap": MAX_APR},
    "expected_profit_on_underpriced_loans": round(float(te.loc[te["margin_vs_break_even"] < 0, "exp_profit"].sum()), 0),
    "n_underpriced": int((te["margin_vs_break_even"] < 0).sum()),
    "overall_pct_underpriced": round(float((te["margin_vs_break_even"] < 0).mean()), 4),
    "by_quintile": price_tbl.reset_index().astype({"pd_quintile": str}).to_dict("records"),
}

# =============================================================================
# 5. Approval cutoff optimisation + LGD sensitivity
# =============================================================================
def cutoff_curve(frame, lgd, n_points=60):
    frame = frame.sort_values("pd").reset_index(drop=True)
    P, r, d = frame["principal"].values, frame["actual_apr"].values, frame["default"].values
    exp_p = np.array([expected_profit(p, a, b, lgd) for p, a, b in zip(frame["pd"], P, r)])
    real_p = realized_profit(d, P, r, lgd)
    rows = []
    for k in np.unique(np.linspace(len(frame) * 0.05, len(frame), n_points).astype(int)):
        rows.append({
            "approval_rate": k / len(frame), "pd_cutoff": frame["pd"].iloc[k - 1],
            "approved_default_rate": d[:k].mean(),
            "expected_profit_total": exp_p[:k].sum(), "realized_profit_total": real_p[:k].sum(),
            "realized_profit_per_loan": real_p[:k].mean(),
        })
    return pd.DataFrame(rows)

curve = cutoff_curve(te, BASE_LGD)
curve.round(4).to_csv(OUT / "cutoff_curve.csv", index=False)
best_real = curve.loc[curve["realized_profit_total"].idxmax()]
best_exp = curve.loc[curve["expected_profit_total"].idxmax()]
all_loans = curve.iloc[-1]
results["cutoff"] = {
    "approve_all": {"approval_rate": 1.0, "default_rate": round(float(all_loans["approved_default_rate"]), 4),
                    "realized_profit_total": round(float(all_loans["realized_profit_total"]), 0)},
    "optimal_by_expected_profit": {k: round(float(best_exp[k]), 4) for k in
                                   ["approval_rate", "pd_cutoff", "approved_default_rate", "realized_profit_total"]},
    "optimal_by_realized_profit": {k: round(float(best_real[k]), 4) for k in
                                   ["approval_rate", "pd_cutoff", "approved_default_rate", "realized_profit_total"]},
}

sens = []
for lgd in LGD_GRID:
    c = cutoff_curve(te, lgd)
    b = c.loc[c["expected_profit_total"].idxmax()]      # decision a lender could make ex ante
    sens.append({"lgd": lgd, "optimal_approval_rate": round(b["approval_rate"], 3),
                 "optimal_pd_cutoff": round(b["pd_cutoff"], 3),
                 "approved_default_rate": round(b["approved_default_rate"], 3),
                 "realized_profit_total": round(b["realized_profit_total"], 0),
                 "realized_profit_approve_all": round(c.iloc[-1]["realized_profit_total"], 0)})
sens = pd.DataFrame(sens)
sens.to_csv(OUT / "lgd_sensitivity.csv", index=False)
results["lgd_sensitivity"] = sens.to_dict("records")

# ---- Model-based selection vs the platform's own underwriting flag ------------
policy_mask = te["credit_policy"] == 1
n_pol = int(policy_mask.sum())
te_sorted = te.sort_values("pd").reset_index(drop=True)
model_sel = te_sorted.iloc[:n_pol]
pol_sel = te[policy_mask]
rate_pol, rate_mod = pol_sel["default"].mean(), model_sel["default"].mean()
cnt = np.array([model_sel["default"].sum(), pol_sel["default"].sum()])
nobs = np.array([len(model_sel), len(pol_sel)])
z, p_z = sm.stats.proportions_ztest(cnt, nobs)
results["hypothesis_tests"]["H3_model_vs_credit_policy_same_approval_rate"] = {
    "approval_rate": round(n_pol / len(te), 4),
    "model_selected_default_rate": round(float(rate_mod), 4),
    "credit_policy_default_rate": round(float(rate_pol), 4),
    "z": round(float(z), 2), "p_value_two_sided": float(p_z),
    "realized_profit_model": round(float(model_sel["real_profit"].sum()), 0),
    "realized_profit_credit_policy": round(float(pol_sel["real_profit"].sum()), 0),
    "note": "credit_policy is a post-underwriting flag and was NOT used as a model feature.",
}

# =============================================================================
# 6. Monitoring: PSI + simulated drift
# =============================================================================
p_ref = lr.predict_proba(X_tr)[:, 1] if champion_is_lr else gb.predict_proba(X_tr)[:, 1]
mon = {"score_psi_train_vs_test": round(psi(p_ref, p_champ), 4)}
feat_psi = {c: round(psi(X_tr[c].values, X_te[c].values), 4) for c in NUM}
mon["feature_psi_train_vs_test"] = feat_psi

# SIMULATED deterioration (synthetic - not observed): lower FICO, higher DTI/utilisation
X_drift = X_te.copy()
X_drift["fico"] = X_drift["fico"] - 30
X_drift["dti"] = X_drift["dti"] + 4
X_drift["revol_util"] = (X_drift["revol_util"] + 12).clip(upper=120)
model_for_drift = lr if champion_is_lr else gb
p_drift = model_for_drift.predict_proba(X_drift)[:, 1]
mon["simulated_drift"] = {
    "scenario": "FICO -30, DTI +4, revolving utilisation +12pts (synthetic)",
    "score_psi": round(psi(p_ref, p_drift), 4),
    "feature_psi": {c: round(psi(X_tr[c].values, X_drift[c].values), 4) for c in ["fico", "dti", "revol_util"]},
    "mean_pd_before": round(float(p_champ.mean()), 4), "mean_pd_after": round(float(p_drift.mean()), 4),
}
psi_flag = lambda v: "stable" if v < 0.10 else ("watch" if v < 0.25 else "investigate")
mon["thresholds"] = {"<0.10": "stable", "0.10-0.25": "watch", ">0.25": "investigate / recalibrate"}
mon["simulated_drift"]["score_psi_status"] = psi_flag(mon["simulated_drift"]["score_psi"])
results["monitoring"] = mon

# =============================================================================
# Figures
# =============================================================================
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(5.5, 5))
for name, p in [("Logistic", p_lr), ("Gradient boosting", p_gb), ("FICO only", p_fico)]:
    fpr, tpr, _ = roc_curve(y_te, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Hold-out ROC")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "roc.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(cal_tbl["avg_pred_pd"], cal_tbl["actual_default"], "o-", label=champ_name)
m = max(cal_tbl["avg_pred_pd"].max(), cal_tbl["actual_default"].max()) * 1.05
ax.plot([0, m], [0, m], "k--", lw=0.8, label="Perfect calibration")
ax.set(xlabel="Mean predicted PD (decile)", ylabel="Actual default rate", title="Calibration by risk decile")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "calibration.png"); plt.close(fig)

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(curve["approval_rate"] * 100, curve["realized_profit_total"] / 1000, color="tab:green",
         label="Realized profit ($k)")
ax1.plot(curve["approval_rate"] * 100, curve["expected_profit_total"] / 1000, color="tab:green",
         ls="--", label="Model-expected profit ($k)")
ax1.axvline(best_exp["approval_rate"] * 100, color="grey", ls=":", lw=1)
ax1.set(xlabel="Approval rate (%) - lowest-PD loans approved first", ylabel="Portfolio profit ($k)")
ax2 = ax1.twinx()
ax2.plot(curve["approval_rate"] * 100, curve["approved_default_rate"] * 100, color="tab:red", label="Default rate of approved (%)")
ax2.set_ylabel("Default rate of approved (%)")
ax2.spines["right"].set_visible(True)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="lower center")
ax1.set_title(f"Approval cutoff trade-off (LGD {int(BASE_LGD*100)}%)")
fig.tight_layout(); fig.savefig(OUT / "cutoff_tradeoff.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
x = np.arange(len(price_tbl)); w = 0.38
ax.bar(x - w / 2, price_tbl["avg_charged_apr"] * 100, w, label="Rate charged")
ax.bar(x + w / 2, price_tbl["avg_break_even_apr"] * 100, w, label=f"Break-even rate (LGD {int(BASE_LGD*100)}%)")
ax.set_xticks(x); ax.set_xticklabels([str(i).replace(" (", "\n(") for i in price_tbl.index], fontsize=8)
ax.set(ylabel="APR (%)", title="Charged vs break-even rate by risk quintile")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "pricing_by_quintile.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4))
names = list(feat_psi) + ["MODEL SCORE"]
vals = list(feat_psi.values()) + [mon["score_psi_train_vs_test"]]
ax.barh(names, vals, color=["tab:blue"] * len(feat_psi) + ["tab:orange"])
ax.axvline(0.10, color="grey", ls=":"); ax.axvline(0.25, color="red", ls=":")
ax.set(xlabel="PSI (train vs hold-out)", title="Population stability (random split => stable)")
fig.tight_layout(); fig.savefig(OUT / "psi.png"); plt.close(fig)

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(json.dumps(results, indent=2, default=float))
