"""
Generate a synthetic dataset of 10,000 loan applicants with no formal credit history
(gig workers, students, thin-file / under-banked individuals).

Each applicant has hidden traits (financial discipline, financial stress, gambling habit,
fraud intent). The visible signals (recharges, bills, UPI activity, apps, device) are
noisy reflections of those traits, and the label `defaulted` is drawn from them.
This mimics real life: the model never sees the hidden traits, only their traces.

Run:  python data/generate_data.py
Out:  data/synthetic_applicants.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

N = 10_000
SEED = 42
OUT = Path(__file__).resolve().parent / "synthetic_applicants.csv"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    rng = np.random.default_rng(SEED)

    # ---------------- hidden traits (never given to the model) ----------------
    discipline = rng.normal(0, 1, N)          # pays on time, plans spending
    stress = rng.normal(0, 1, N)              # money pressure
    gambler = rng.random(N) < sigmoid(-2.2 + 0.8 * stress - 0.6 * discipline)
    fraud_intent = rng.random(N) < 0.06       # small group trying to game the system

    # ---------------- profile ----------------
    segment = rng.choice(
        ["gig_worker", "student", "thin_file", "informal_salaried"],
        size=N, p=[0.35, 0.25, 0.25, 0.15],
    )
    age = np.where(segment == "student", rng.integers(18, 26, N),
          np.where(segment == "gig_worker", rng.integers(20, 46, N),
                   rng.integers(21, 56, N)))
    city_tier = rng.choice([1, 2, 3], size=N, p=[0.35, 0.40, 0.25])
    base_income = {"gig_worker": 22000, "student": 8000, "thin_file": 18000, "informal_salaried": 25000}
    monthly_income = np.array([base_income[s] for s in segment]) * rng.lognormal(0, 0.35, N)
    monthly_income = monthly_income * np.where(city_tier == 1, 1.2, np.where(city_tier == 2, 1.0, 0.8))
    income_variability = np.clip(
        np.where(segment == "gig_worker", rng.beta(4, 5, N), rng.beta(2, 7, N)), 0, 1)
    employment_months = np.where(segment == "student", 0, rng.gamma(2.0, 14, N)).round()
    dependents = rng.poisson(np.where(segment == "student", 0.1, 1.2))

    # ---------------- telecom ----------------
    recharge_on_time_ratio = sigmoid(1.6 + 1.2 * discipline + rng.normal(0, 0.6, N))
    phone_bill_on_time_ratio = sigmoid(1.4 + 1.1 * discipline + rng.normal(0, 0.7, N))
    recharge_gaps_6m = rng.poisson(np.exp(0.2 - 0.8 * discipline + 0.3 * stress))
    avg_recharge_amount = np.clip(monthly_income * 0.012 * rng.lognormal(0, 0.3, N), 99, 999).round()
    sim_age_months = np.where(fraud_intent, rng.gamma(1.2, 3, N), rng.gamma(2.5, 18, N)).round()
    sim_changes_12m = rng.poisson(np.where(fraud_intent, 2.5, 0.3))

    # ---------------- utility bills ----------------
    utility_bill_on_time_ratio = sigmoid(1.3 + 1.3 * discipline + rng.normal(0, 0.7, N))
    rent_on_time_ratio = sigmoid(1.5 + 1.0 * discipline - 0.4 * stress + rng.normal(0, 0.7, N))

    # ---------------- UPI behaviour ----------------
    upi_monthly_inflow = monthly_income * rng.normal(1.0, 0.12, N).clip(0.6, 1.5)
    spend_ratio = np.clip(0.82 + 0.12 * stress - 0.08 * discipline + rng.normal(0, 0.08, N), 0.4, 1.6)
    upi_monthly_spend = upi_monthly_inflow * spend_ratio
    upi_txn_count_month = rng.poisson(np.clip(monthly_income / 600, 8, 120))
    pct_spend_gambling = np.where(gambler, rng.beta(2, 10, N), rng.beta(0.3, 60, N))
    pct_spend_entertainment = rng.beta(2, 14, N)
    pct_spend_essentials = np.clip(1 - pct_spend_gambling - pct_spend_entertainment - rng.beta(3, 8, N), 0.05, 0.95)
    p2p_txn_ratio = rng.beta(3, 5, N)
    failed_txn_ratio = sigmoid(-3.0 + 0.7 * stress + rng.normal(0, 0.4, N))
    late_night_txn_ratio = np.where(gambler, rng.beta(4, 8, N), rng.beta(2, 14, N))
    avg_balance = np.clip(monthly_income * rng.lognormal(-1.2 + 0.5 * discipline - 0.5 * stress, 0.5, N), 0, None)
    low_balance_days_month = rng.binomial(30, sigmoid(-1.3 + 0.9 * stress - 0.6 * discipline))

    # ---------------- installed apps ----------------
    num_gambling_apps = np.where(gambler, rng.poisson(1.8, N) + 1, rng.poisson(0.05, N))
    num_loan_apps = rng.poisson(np.exp(-0.6 + 0.8 * stress))
    num_finance_apps = rng.poisson(np.exp(0.6 + 0.3 * discipline))
    num_apps_total = rng.poisson(60, N) + num_gambling_apps + num_loan_apps + num_finance_apps

    # ---------------- device & identity ----------------
    device_type = rng.choice(["phone", "tablet", "tv"], size=N, p=[0.94, 0.05, 0.01])
    os_version_age_months = rng.gamma(2, 8, N).round()
    location_consistency_score = np.where(fraud_intent, rng.beta(3, 4, N), rng.beta(9, 2, N))

    def flag(p_normal, p_fraud):
        return (rng.random(N) < np.where(fraud_intent, p_fraud, p_normal)).astype(int)

    device_rooted = flag(0.015, 0.35)
    vpn_active = flag(0.03, 0.40)
    emulator_detected = flag(0.003, 0.25)
    vm_detected = flag(0.002, 0.15)
    frida_detected = flag(0.001, 0.20)
    timezone_mismatch = flag(0.01, 0.40)     # device vs SIM vs network timezone
    device_id_mismatch = flag(0.005, 0.30)   # MAC / IMEI does not match registered device
    device_type = np.where(fraud_intent & (rng.random(N) < 0.08), "tv", device_type)

    # ---------------- label: did the person default? ----------------
    logit = (
        -2.3
        - 1.0 * discipline
        + 0.8 * stress
        + 1.0 * gambler
        + 0.6 * (spend_ratio > 1.0)
        - 0.35 * np.log(monthly_income / 15000)
        + 0.8 * income_variability
        + 2.5 * fraud_intent
        + rng.normal(0, 0.4, N)
    )
    defaulted = (rng.random(N) < sigmoid(logit)).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"A{i:05d}" for i in range(N)],
        "segment": segment, "age": age, "city_tier": city_tier,
        "monthly_income": monthly_income.round(), "income_variability": income_variability.round(3),
        "employment_months": employment_months, "dependents": dependents,
        "recharge_on_time_ratio": recharge_on_time_ratio.round(3),
        "phone_bill_on_time_ratio": phone_bill_on_time_ratio.round(3),
        "recharge_gaps_6m": recharge_gaps_6m, "avg_recharge_amount": avg_recharge_amount,
        "sim_age_months": sim_age_months, "sim_changes_12m": sim_changes_12m,
        "utility_bill_on_time_ratio": utility_bill_on_time_ratio.round(3),
        "rent_on_time_ratio": rent_on_time_ratio.round(3),
        "upi_monthly_inflow": upi_monthly_inflow.round(), "upi_monthly_spend": upi_monthly_spend.round(),
        "upi_txn_count_month": upi_txn_count_month,
        "pct_spend_essentials": pct_spend_essentials.round(3),
        "pct_spend_entertainment": pct_spend_entertainment.round(3),
        "pct_spend_gambling": pct_spend_gambling.round(3),
        "p2p_txn_ratio": p2p_txn_ratio.round(3), "failed_txn_ratio": failed_txn_ratio.round(3),
        "late_night_txn_ratio": late_night_txn_ratio.round(3),
        "avg_balance": avg_balance.round(), "low_balance_days_month": low_balance_days_month,
        "num_gambling_apps": num_gambling_apps, "num_loan_apps": num_loan_apps,
        "num_finance_apps": num_finance_apps, "num_apps_total": num_apps_total,
        "device_type": device_type, "os_version_age_months": os_version_age_months,
        "location_consistency_score": location_consistency_score.round(3),
        "device_rooted": device_rooted, "vpn_active": vpn_active,
        "emulator_detected": emulator_detected, "vm_detected": vm_detected,
        "frida_detected": frida_detected, "timezone_mismatch": timezone_mismatch,
        "device_id_mismatch": device_id_mismatch,
        "defaulted": defaulted,
    })
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} applicants to {OUT}")
    print(f"Default rate: {df['defaulted'].mean():.1%}")


if __name__ == "__main__":
    main()
