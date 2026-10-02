# Consumer Credit Risk & Risk-Based Pricing Model

An end-to-end consumer-lending analytics project on public LendingClub loan data (9,578 loans, 2007-2010):
predict **probability of default (PD)**, validate the model, then use it to answer the business questions a lending
analytics team actually owns: **who to approve, what to charge, and how to monitor the model once it is live.**

| Question | What this project does |
|---|---|
| Can we rank borrowers by default risk? | Logistic regression vs gradient boosting vs a FICO-only baseline, scored on a held-out 30% |
| Is the model trustworthy? | AUC / Gini / KS, calibration by decile, bootstrap confidence intervals |
| What actually drives default? | Hypothesis tests (chi-square, logit coefficient tests, two-proportion z-test) |
| What should we charge? | Break-even APR for every loan given its PD, LGD and funding cost; share of loans under-priced |
| Who should we approve? | Approval-cutoff optimisation: approval rate vs loss rate vs profit, with LGD sensitivity |
| How do we keep it healthy? | PSI monitoring on score and features, plus a simulated-drift stress test and a monitoring plan |

## Key results (hold-out set of 2,874 loans, seed 42)

**Model performance**

| Model | Test AUC | Gini | KS |
|---|---|---|---|
| Logistic regression (champion) | 0.656 | 0.312 | 0.215 |
| Gradient boosting | 0.661 | 0.322 | 0.239 |
| Baseline: FICO only | 0.623 | 0.245 | 0.169 |

- Adding the other borrower attributes lifts AUC from 0.623 (FICO alone) to 0.656.
- Gradient boosting is **not meaningfully better**: the 95% bootstrap CI on the AUC difference is [-0.008, +0.017], which includes zero. The interpretable logistic model is therefore the champion.
- Calibration is good: mean predicted PD 16.0% vs actual 16.0%. The riskiest decile defaults at 35.1% vs 4.9% in the safest (about **7x separation**).

**What drives default (hypothesis tests)**

- Default rate falls monotonically with FICO band: 30.9% (<660), 19.4% (660-699), 15.3% (700-739), 8.1% (740+). Chi-square = 216.4, p < 1e-45.
- After controlling for the other features, the strongest drivers are **recent credit inquiries** (odds ratio 1.36 per 1 SD, p < 1e-22), **FICO** (0.69 per SD, p < 1e-13) and **payment-to-income** (1.28 per SD, p < 1e-10). Revolving utilisation, DTI, income, delinquencies and public records are **not** statistically significant at 5% once these are in the model.

**Pricing (assumptions below)**

- Average charged APR vs break-even APR by risk quintile: Q1 9.3% vs 6.5%, Q3 12.6% vs 9.8%, Q5 (riskiest) **14.6% vs 17.5%**.
- **71% of loans in the riskiest quintile are priced below break-even**, and that quintile loses an average of about $274 per loan in the backtest. Across the whole hold-out set, 20.7% of loans (594) are under-priced, with a combined expected loss of about $240K.

**Approval cutoff (assumptions below)**

- Approving everyone yields about **$553K** backtested profit. Choosing the cutoff by *expected* profit (an ex-ante decision, no peeking at outcomes) means approving the lowest-PD 83.9% of applicants (PD cutoff 22.7%), which yields about **$803K (+45%)** while cutting the approved default rate from 16.0% to 13.2%.
- The optimal cutoff is very sensitive to loss severity: at LGD 30% the optimum is to approve 96.8%; at LGD 70% it is 58.1%, and approving everyone turns loss-making. **LGD is the assumption to pressure-test first.**

**An honest negative result**

- At the same approval rate (80.4%), selecting loans with this model gives a 13.1% default rate vs 13.3% for LendingClub's own `credit.policy` flag (two-proportion z-test p = 0.83): **no statistically significant improvement**. The flag was deliberately excluded from the model as a feature because it is a post-underwriting decision. The model reproduces the platform's underwriting quality from borrower attributes alone; it does not beat it.

**Monitoring**

- Train-vs-hold-out PSI is 0.004 for the score and below 0.01 for every feature (stable, as expected from a random split).
- In a **simulated** deterioration (FICO -30, DTI +4, utilisation +12 pts; synthetic, not observed), score PSI rises to 0.31, above the 0.25 "investigate" threshold, and mean predicted PD rises from 16.0% to 20.9%.

## Monitoring plan (what I would run in production)

| Metric | Cadence | Green | Amber | Red / action |
|---|---|---|---|---|
| Score PSI | Monthly | < 0.10 | 0.10-0.25 | > 0.25: investigate, check data pipeline and population mix |
| Feature PSI (FICO, DTI, inquiries, utilisation) | Monthly | < 0.10 | 0.10-0.25 | > 0.25: root-cause which segment shifted |
| Calibration (predicted vs actual PD by decile) | Quarterly, once outcomes mature | within +/- 20% relative | 20-35% | > 35%: recalibrate |
| Rank-ordering (AUC / KS on matured cohorts) | Quarterly | within 0.03 of validation | 0.03-0.05 | > 0.05 drop: redevelop |
| Under-priced share by risk band | Monthly | stable | rising 3+ months | reprice or tighten cutoff for that band |

Thresholds are conventional industry rules of thumb, not fitted to this data.

## Assumptions (not estimated from the data - change them in `src/credit_risk.py`)

The dataset has **no loan term, loan amount, recovery data or dates**, so the economics rest on stated assumptions:

- **Term:** 36 months. Principal is backed out from installment and rate.
- **LGD:** 50% of original principal, net of payments before default (sensitivity: 30-70%).
- **Funding + servicing cost:** 4% per year on average outstanding balance.
- **APR cap** of 36% when solving for break-even rates.
- "Realized profit" means *backtested profit under these assumptions using actual default outcomes*. It is **not** actual lender P&L.

## Limitations

- **Modest discrimination (AUC 0.66).** The dataset is a small cleaned extract with few features. Bureau-depth variables would be needed to go much further.
- **No time dimension.** There are no dates, so validation is a stratified random hold-out plus 5-fold CV, not an out-of-time test. A production version must validate out-of-time.
- **Selection bias.** All loans were already accepted by LendingClub, so the model is trained only on approved applicants (reject inference is out of scope). `int_rate` is excluded as a feature because LendingClub set it using its own risk view, which would make the pricing analysis circular.
- Drift scenario is synthetic by design.

## Method notes

- **Features:** FICO, DTI, log income, credit history length, log revolving balance, revolving utilisation, inquiries (6m), delinquencies (2y), public records, payment-to-income (annual installment / income), loan purpose (one-hot).
- **Excluded on purpose:** `credit.policy` (post-underwriting decision) and `int.rate` (set by the lender using its own risk view).
- **Tuning:** 5-fold stratified CV grid search on the training set only; the hold-out set is touched once for reporting.
- **Break-even APR:** solved per loan with Brent's method so that `(1 - PD) x interest - PD x LGD x principal - funding cost = 0`.
- **Cutoff:** loans sorted by predicted PD, approved from lowest risk upward; the optimal cutoff is chosen on *expected* profit, then evaluated on realized outcomes.

## Run it

```bash
pip install -r requirements.txt
python src/credit_risk.py
```

The script downloads the dataset automatically if `data/loan_data.csv` is missing, runs in well under a minute, and writes tables, charts and `results.json` to `outputs/`. All results above come from this script (fixed seed, reproducible).

## Repository layout

```
src/credit_risk.py   full pipeline (modeling, validation, pricing, cutoff, monitoring, tests)
data/                loan_data.csv (auto-downloaded)
outputs/             results.json, CSV tables, ROC / calibration / cutoff / pricing / PSI charts
requirements.txt
```

## Data

Public LendingClub loan data (2007-2010), cleaned extract widely mirrored for education. Each row is a loan with borrower attributes and a flag for whether it was not fully paid (16.0% of loans). Original source: LendingClub public data. This repository uses it for educational and portfolio purposes only.
