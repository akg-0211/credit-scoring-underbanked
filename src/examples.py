"""Ready-made applicants to try on the screen (and in tests)."""

GOOD = {
    "segment": "gig_worker", "age": 27, "city_tier": 2,
    "monthly_income": 26000, "income_variability": 0.25, "employment_months": 30, "dependents": 1,
    "recharge_on_time_ratio": 0.97, "phone_bill_on_time_ratio": 0.95, "recharge_gaps_6m": 0,
    "avg_recharge_amount": 299, "sim_age_months": 48, "sim_changes_12m": 0,
    "utility_bill_on_time_ratio": 0.96, "rent_on_time_ratio": 0.95,
    "upi_monthly_inflow": 27000, "upi_monthly_spend": 19500, "upi_txn_count_month": 45,
    "pct_spend_essentials": 0.72, "pct_spend_entertainment": 0.08, "pct_spend_gambling": 0.0,
    "p2p_txn_ratio": 0.35, "failed_txn_ratio": 0.02, "late_night_txn_ratio": 0.05,
    "avg_balance": 11000, "low_balance_days_month": 2,
    "num_gambling_apps": 0, "num_loan_apps": 0, "num_finance_apps": 3, "num_apps_total": 62,
    "device_type": "phone", "os_version_age_months": 10, "location_consistency_score": 0.92,
    "device_rooted": 0, "vpn_active": 0, "emulator_detected": 0, "vm_detected": 0,
    "frida_detected": 0, "timezone_mismatch": 0, "device_id_mismatch": 0,
}

RISKY = {
    **GOOD,
    "segment": "student", "age": 21, "monthly_income": 9000, "employment_months": 0, "dependents": 0,
    "income_variability": 0.45,
    "recharge_on_time_ratio": 0.8, "phone_bill_on_time_ratio": 0.75, "recharge_gaps_6m": 3,
    "utility_bill_on_time_ratio": 0.7, "rent_on_time_ratio": 0.55,
    "upi_monthly_inflow": 9500, "upi_monthly_spend": 9800, "upi_txn_count_month": 30,
    "pct_spend_gambling": 0.07, "pct_spend_entertainment": 0.15, "pct_spend_essentials": 0.55,
    "late_night_txn_ratio": 0.4, "avg_balance": 900, "low_balance_days_month": 16,
    "num_gambling_apps": 2, "num_loan_apps": 3, "failed_txn_ratio": 0.08,
}

GREY = {
    **GOOD,
    "num_gambling_apps": 1, "pct_spend_gambling": 0.02, "recharge_gaps_6m": 3,
    "sim_changes_12m": 2, "num_loan_apps": 1, "upi_monthly_spend": 23500, "avg_balance": 6000,
}

RED = {**GOOD, "device_rooted": 1, "vpn_active": 1}

EXAMPLES = {"good": GOOD, "grey": GREY, "risky": RISKY, "red": RED}
