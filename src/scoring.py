"""
Scoring engine: rules + model -> one JSON decision.

Flow for one applicant:
  1. RED check   - any device/identity red flag  -> rejected outright (score 0).
  2. Model       - Random Forest gives P(default); base score = 100 * (1 - P(default)).
  3. GREY check  - policy rules subtract penalty points by severity.
  4. Decision    - 90+ GREAT, 85-90 OK, 80-85 NOT GREAT, below 80 NOT ELIGIBLE.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.features import RED_FLAG_FIELDS, engineer, to_model_matrix

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

THRESHOLDS = {"GREAT": 90, "OK": 85, "NOT GREAT": 80}

RED_FLAG_TEXT = {
    "device_rooted": "Device is rooted",
    "vpn_active": "VPN is active during application",
    "emulator_detected": "App is running on an emulator",
    "vm_detected": "App is running inside a virtual machine",
    "frida_detected": "Frida / hooking tool detected",
    "timezone_mismatch": "Device, SIM and network timezones do not match",
    "device_id_mismatch": "Device IDs (MAC / IMEI) do not match the registered device",
}

# Signals that explain a weak model score. direction +1: higher is better, -1: lower is better
WEAK_SIGNAL_TEXT = {
    "savings_rate": (+1, "Saves very little of monthly inflow"),
    "low_balance_share": (-1, "Account balance is low on many days of the month"),
    "bill_payment_score": (+1, "Bills and recharges are often paid late"),
    "balance_to_income": (+1, "Average balance is small compared to income"),
    "spend_to_inflow_ratio": (-1, "Spends almost all (or more than) the money coming in"),
    "failed_txn_ratio": (-1, "Many failed UPI transactions"),
    "income_stability": (+1, "Income is very irregular"),
    "recharge_on_time_ratio": (+1, "Mobile recharges are often late"),
}


class CreditScorer:
    def __init__(self, models_dir: Path = MODELS):
        self.model = joblib.load(models_dir / "model.joblib")
        self.columns = json.loads((models_dir / "feature_columns.json").read_text())
        self.reference = json.loads((models_dir / "reference_ranges.json").read_text())
        self.metrics = json.loads((models_dir / "metrics.json").read_text())

    # ---------------- step 1: RED ----------------
    @staticmethod
    def red_flags(a: dict) -> list[str]:
        flags = [RED_FLAG_TEXT[f] for f in RED_FLAG_FIELDS if int(a.get(f, 0)) == 1]
        if a.get("device_type") == "tv":
            flags.append("Applying from a TV / non-personal device")
        return flags

    # ---------------- step 3: GREY ----------------
    @staticmethod
    def grey_flags(a: dict, f: pd.Series) -> list[dict]:
        out = []

        def add(reason, penalty):
            out.append({"reason": reason, "penalty": penalty})

        if a["num_gambling_apps"] >= 1:
            add(f"{int(a['num_gambling_apps'])} gambling app(s) installed",
                min(2 * int(a["num_gambling_apps"]), 8))
        if a["pct_spend_gambling"] > 0.05:
            add(f"{a['pct_spend_gambling']:.0%} of UPI spend goes to gambling", 6)
        elif a["pct_spend_gambling"] > 0.01:
            add(f"{a['pct_spend_gambling']:.0%} of UPI spend goes to gambling", 3)
        if a["num_loan_apps"] >= 3:
            add(f"{int(a['num_loan_apps'])} loan apps installed (possible loan stacking)", 5)
        if f["worst_bill_ratio"] < 0.6:
            add("At least one bill type is paid late more than 40% of the time", 5)
        elif f["bill_payment_score"] < 0.75:
            add("Overall bill payment record is below 75% on time", 3)
        if a["recharge_gaps_6m"] >= 3:
            add(f"{int(a['recharge_gaps_6m'])} gaps in mobile recharge in 6 months", 2)
        if f["spend_to_inflow_ratio"] > 1.0:
            add("Monthly UPI spend is higher than monthly inflow", 5)
        if a["failed_txn_ratio"] > 0.10:
            add(f"{a['failed_txn_ratio']:.0%} of UPI transactions fail", 3)
        if a["sim_changes_12m"] >= 2:
            add(f"SIM changed {int(a['sim_changes_12m'])} times in 12 months", 3)
        if a["location_consistency_score"] < 0.5:
            add("Location pattern is inconsistent", 4)
        if a["late_night_txn_ratio"] > 0.35:
            add(f"{a['late_night_txn_ratio']:.0%} of transactions happen late at night", 2)
        if a["low_balance_days_month"] >= 15:
            add(f"Balance is near zero on {int(a['low_balance_days_month'])} days a month", 3)
        return out

    def weak_signals(self, f: pd.Series) -> list[str]:
        reasons = []
        for col, (direction, text) in WEAK_SIGNAL_TEXT.items():
            ref = self.reference[col]
            if (direction > 0 and f[col] < ref["p10"]) or (direction < 0 and f[col] > ref["p90"]):
                reasons.append(text)
        return reasons

    @staticmethod
    def decision_for(score: float) -> str:
        for label, cut in THRESHOLDS.items():
            if score >= cut:
                return label
        return "NOT ELIGIBLE"

    # ---------------- full pipeline ----------------
    def score(self, applicant: dict) -> dict:
        reds = self.red_flags(applicant)
        if reds:
            return {
                "score": 0,
                "category": "RED",
                "decision": "NOT ELIGIBLE",
                "eligible": False,
                "rejection_confidence": "100%",
                "red_flags": reds,
                "grey_flags": [],
                "reasons": ["Rejected by fraud/device rules: " + "; ".join(reds)],
                "model": None,
                "thresholds": THRESHOLDS,
            }

        raw = pd.DataFrame([applicant])
        feats = engineer(raw).iloc[0]
        X = to_model_matrix(raw, self.columns)
        p_default = float(self.model.predict_proba(X)[0, 1])
        base_score = round(100 * (1 - p_default), 1)

        greys = self.grey_flags(applicant, feats)
        penalty = sum(g["penalty"] for g in greys)
        final = round(max(0.0, min(100.0, base_score - penalty)), 1)
        decision = self.decision_for(final)

        reasons = [g["reason"] for g in greys]
        if base_score < THRESHOLDS["NOT GREAT"]:
            reasons += self.weak_signals(feats) or ["Overall money behaviour looks risky to the model"]

        return {
            "score": final,
            "category": "GREY" if greys else "WHITE",
            "decision": decision,
            "eligible": decision != "NOT ELIGIBLE",
            "red_flags": [],
            "grey_flags": greys,
            "reasons": reasons,
            "model": {
                "default_probability": round(p_default, 4),
                "base_score": base_score,
                "grey_penalty": penalty,
            },
            "thresholds": THRESHOLDS,
        }


if __name__ == "__main__":
    import sys
    from src.examples import EXAMPLES

    name = sys.argv[1] if len(sys.argv) > 1 else "good"
    print(json.dumps(CreditScorer().score(EXAMPLES[name]), indent=2))
