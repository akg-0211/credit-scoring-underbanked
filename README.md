# Alternative Credit Scoring for Under-banked Individuals

Gig workers, students and people with no banking history are usually refused loans because they have no credit score. This project scores them using the signals they *do* have: mobile recharges, bill payments, UPI behaviour, installed apps, and device integrity.

The output is a 0–100 score, a risk category (WHITE / GREY / RED), a decision, and the reasons behind it, as JSON. A web screen lets you enter any custom applicant and see the result instantly.

## How a decision is made

```
Applicant data
     │
     ▼
1. RED check (rules) ── any device/identity red flag ──► REJECTED (score 0, 100% reject)
     │ no red flags
     ▼
2. Model: Random Forest on 80 engineered features
   base score = 100 × (1 − P(default))
     │
     ▼
3. GREY check (policy rules) ── subtract penalty points by severity
     │
     ▼
4. Decision:  90+ GREAT · 85–90 OK · 80–85 NOT GREAT · below 80 NOT ELIGIBLE
```

| Category | Meaning |
|---|---|
| **RED** | Rooted device, VPN, emulator, virtual machine, Frida, timezone mismatch (device / SIM / network), device ID (MAC / IMEI) mismatch, or a TV / non-personal device. Rejected outright. |
| **GREY** | Risk signals that lower the score: gambling apps or gambling spend, loan-app stacking, late bills, recharge gaps, spending more than inflow, failed transactions, frequent SIM changes, inconsistent location, heavy late-night spending, many low-balance days. |
| **WHITE** | No red or grey signals; score comes from the model alone. |

## Data

`data/generate_data.py` creates **10,000 synthetic applicants** across four segments (gig workers, students, thin-file users, informal salaried). Each applicant has hidden traits (financial discipline, money stress, gambling habit, fraud intent). The visible signals are noisy reflections of those traits, and the `defaulted` label is drawn from them, so the model has to learn from indirect evidence, as it would in real life.

## Features

80 engineered features, including:
- **Raw signals:** income, income variability, on-time ratios for recharges / phone bill / utilities / rent, UPI inflow and spend, spend mix, failed and late-night transactions, balances, app counts, SIM age and changes, location consistency.
- **Ratios:** spend-to-inflow, savings rate, balance-to-income, income per dependent, bill payment score, worst bill ratio, gambling spend amount, finance-app share, SIM stability.
- **Behaviour flags:** overspender, has gambling app, loan-app stacking, late bill payer, frequent SIM changes, night-owl spender.
- **One-hot encoded bands:** segment, city tier, device type, age band, income band, work-tenure band, gambling level, bill discipline.

Device red-flag fields are deliberately *not* model features. They are handled by hard rules, and red-flagged applicants (1,251 of 10,000) are excluded from training.

## Results (5-fold stratified cross-validation, 8,749 applicants)

| Metric | Value |
|---|---|
| Accuracy | 83.0% |
| Baseline accuracy (always predict "repays") | 78.7% |
| ROC-AUC | 0.815 |
| Precision (defaulters) | 0.69 |
| Recall (defaulters) | 0.37 |

Accuracy alone is misleading here because most applicants repay; ROC-AUC shows how well the model ranks risky vs safe applicants.

**Do the score bands mean something?** Out-of-fold default rate per band:

| Score band | Applicants | Actually defaulted |
|---|---|---|
| 90+ GREAT | 3,217 | 4.9% |
| 85–90 OK | 1,274 | 11.3% |
| 80–85 NOT GREAT | 895 | 17.1% |
| below 80 NOT ELIGIBLE | 3,363 | 41.8% |

Each step down in band roughly doubles the real default rate, so the thresholds separate risk as intended.

Most important features: low-balance days, savings rate, spend-to-inflow ratio, balance-to-income, and bill payment record.

## Run it

```bash
pip install -r requirements.txt
python data/generate_data.py      # 10,000 synthetic applicants
python -m src.train               # 5-fold CV + final model
uvicorn app.main:app --reload     # open http://localhost:8000
python -m pytest                  # tests
```

Or with Docker (data generation and training happen during the build):

```bash
docker build -t credit-scoring .
docker run -p 8000:8000 credit-scoring
```

## API

`POST /api/score` with an applicant JSON (see `src/examples.py`, or the interactive docs at `/docs`):

```json
{
  "score": 83.7,
  "category": "GREY",
  "decision": "NOT GREAT",
  "eligible": true,
  "grey_flags": [
    {"reason": "1 gambling app(s) installed", "penalty": 2},
    {"reason": "2% of UPI spend goes to gambling", "penalty": 3},
    {"reason": "3 gaps in mobile recharge in 6 months", "penalty": 2},
    {"reason": "SIM changed 2 times in 12 months", "penalty": 3}
  ],
  "model": {"default_probability": 0.0634, "base_score": 93.7, "grey_penalty": 10}
}
```

## Design choices

- **Rules + model, not model alone.** Fraud signals (rooted device, emulator, VPN) are rare and must always reject, so they are hard rules. Behaviour risk is learned by the model. GREY penalties act as transparent policy overlays a lender can tune.
- **No class re-weighting.** The model's probability becomes the score, so probabilities must stay realistic. Re-weighting improved recall but pushed every score down and broke the thresholds.
- **Explainable output.** Every decision lists its reasons, so an applicant can be told why they were refused.

## Limitations

- The data is synthetic. Metrics show the pipeline works; they do not prove real-world accuracy. Real deployment needs real repayment data and consent for every signal used.
- GREY penalties are hand-set policy values, not learned.
- Fairness across segments (students vs gig workers, city tiers) should be audited before any real use.

## Project structure

```
data/generate_data.py   synthetic applicants
src/features.py         feature engineering + one-hot encoding
src/train.py            5-fold CV, metrics, final model
src/scoring.py          RED / model / GREY / decision pipeline
src/examples.py         sample applicants
app/main.py             FastAPI app + API
app/static/index.html   screen for custom applicants
tests/                  pytest tests
```
