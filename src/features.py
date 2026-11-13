"""
Feature engineering shared by training and scoring.

Raw applicant fields go in, a fixed model matrix (70+ columns) comes out.
Using the same function in both places guarantees the model sees identical
columns at training time and at scoring time.
"""
import numpy as np
import pandas as pd

# Device / identity signals. These are NOT model features: any one of them is a
# hard RED flag handled by rules in scoring.py.
RED_FLAG_FIELDS = [
    "device_rooted", "vpn_active", "emulator_detected", "vm_detected",
    "frida_detected", "timezone_mismatch", "device_id_mismatch",
]

CATEGORICAL_FIELDS = ["segment", "city_tier", "device_type"]

NUMERIC_FIELDS = [
    "age", "monthly_income", "income_variability", "employment_months", "dependents",
    "recharge_on_time_ratio", "phone_bill_on_time_ratio", "recharge_gaps_6m",
    "avg_recharge_amount", "sim_age_months", "sim_changes_12m",
    "utility_bill_on_time_ratio", "rent_on_time_ratio",
    "upi_monthly_inflow", "upi_monthly_spend", "upi_txn_count_month",
    "pct_spend_essentials", "pct_spend_entertainment", "pct_spend_gambling",
    "p2p_txn_ratio", "failed_txn_ratio", "late_night_txn_ratio",
    "avg_balance", "low_balance_days_month",
    "num_gambling_apps", "num_loan_apps", "num_finance_apps", "num_apps_total",
    "os_version_age_months", "location_consistency_score",
]

INPUT_FIELDS = CATEGORICAL_FIELDS + NUMERIC_FIELDS + RED_FLAG_FIELDS


def _bucket(series, bins, labels):
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(str)


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw applicant rows into model features (before one-hot alignment)."""
    df = raw.copy()
    eps = 1.0
    f = pd.DataFrame(index=df.index)

    # 1) raw numeric signals
    for col in NUMERIC_FIELDS:
        f[col] = df[col].astype(float)

    # 2) engineered ratios and scores
    f["spend_to_inflow_ratio"] = df["upi_monthly_spend"] / (df["upi_monthly_inflow"] + eps)
    f["savings_rate"] = (df["upi_monthly_inflow"] - df["upi_monthly_spend"]) / (df["upi_monthly_inflow"] + eps)
    f["balance_to_income"] = df["avg_balance"] / (df["monthly_income"] + eps)
    f["income_per_dependent"] = df["monthly_income"] / (df["dependents"] + 1)
    f["income_stability"] = 1 - df["income_variability"]
    bills = df[["recharge_on_time_ratio", "phone_bill_on_time_ratio",
                "utility_bill_on_time_ratio", "rent_on_time_ratio"]]
    f["bill_payment_score"] = bills.mean(axis=1)
    f["worst_bill_ratio"] = bills.min(axis=1)
    f["recharge_to_income"] = df["avg_recharge_amount"] / (df["monthly_income"] + eps)
    f["avg_txn_value"] = df["upi_monthly_spend"] / (df["upi_txn_count_month"] + eps)
    f["gambling_spend_amount"] = df["upi_monthly_spend"] * df["pct_spend_gambling"]
    f["discretionary_share"] = df["pct_spend_gambling"] + df["pct_spend_entertainment"]
    f["risky_app_count"] = df["num_gambling_apps"] + df["num_loan_apps"]
    f["finance_app_share"] = df["num_finance_apps"] / (df["num_apps_total"] + eps)
    f["low_balance_share"] = df["low_balance_days_month"] / 30
    f["sim_stability"] = df["sim_age_months"] / (df["sim_changes_12m"] + 1)

    # 3) binary behaviour flags
    f["overspender_flag"] = (f["spend_to_inflow_ratio"] > 1.0).astype(int)
    f["has_gambling_app"] = (df["num_gambling_apps"] > 0).astype(int)
    f["loan_app_stacking"] = (df["num_loan_apps"] >= 3).astype(int)
    f["late_bill_payer"] = (f["worst_bill_ratio"] < 0.6).astype(int)
    f["frequent_sim_changes"] = (df["sim_changes_12m"] >= 2).astype(int)
    f["night_owl_spender"] = (df["late_night_txn_ratio"] > 0.35).astype(int)
    f["has_income_history"] = (df["employment_months"] > 0).astype(int)

    # 4) bucketed versions (one-hot encoded below)
    cats = pd.DataFrame(index=df.index)
    cats["segment"] = df["segment"].astype(str)
    cats["city_tier"] = df["city_tier"].astype(str)
    cats["device_type"] = df["device_type"].astype(str)
    cats["age_band"] = _bucket(df["age"], [0, 22, 30, 40, 120], ["18-22", "23-30", "31-40", "41+"])
    cats["income_band"] = _bucket(df["monthly_income"], [0, 10000, 18000, 28000, 45000, 1e9],
                                  ["<10k", "10-18k", "18-28k", "28-45k", "45k+"])
    cats["tenure_band"] = _bucket(df["employment_months"], [-1, 0, 12, 36, 1e6],
                                  ["none", "<1y", "1-3y", "3y+"])
    cats["gambling_level"] = _bucket(df["pct_spend_gambling"], [-1, 0.01, 0.08, 1.0],
                                     ["none", "low", "high"])
    cats["bill_discipline"] = _bucket(f["bill_payment_score"], [-1, 0.6, 0.85, 1.0],
                                      ["poor", "fair", "good"])
    one_hot = pd.get_dummies(cats, prefix_sep="=", dtype=int)

    return pd.concat([f, one_hot], axis=1)


def to_model_matrix(raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Engineer features and align to the exact training columns."""
    feats = engineer(raw)
    return feats.reindex(columns=columns, fill_value=0).astype(float)
