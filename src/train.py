"""
Train the repayment-risk model with 5-fold stratified cross-validation.

Applicants with any RED flag are left out of training: they are rejected by
rules before the model is ever asked (see scoring.py).

Run:  python -m src.train
Out:  models/model.joblib, models/feature_columns.json, models/metrics.json
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from src.features import RED_FLAG_FIELDS, engineer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "synthetic_applicants.csv"
MODELS = ROOT / "models"


def main():
    df = pd.read_csv(DATA)
    red = (df[RED_FLAG_FIELDS].sum(axis=1) > 0) | (df["device_type"] == "tv")
    train_df = df[~red].reset_index(drop=True)
    print(f"Loaded {len(df):,} applicants; {red.sum():,} have RED flags (rule-rejected), "
          f"{len(train_df):,} used for the model")

    X = engineer(train_df)
    y = train_df["defaulted"]
    print(f"Engineered features: {X.shape[1]}")

    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
        # No class re-weighting: we turn the predicted probability directly into a
        # 0-100 score, so the probabilities must stay realistic (calibrated).
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y, cv=cv,
                            scoring=["accuracy", "roc_auc", "precision", "recall", "f1"])
    metrics = {k.replace("test_", ""): round(float(v.mean()), 4)
               for k, v in scores.items() if k.startswith("test_")}
    metrics["baseline_accuracy_always_repay"] = round(float(1 - y.mean()), 4)
    metrics["n_features"] = int(X.shape[1])
    metrics["n_train_rows"] = int(len(train_df))

    # Out-of-fold scores: check that the score bands really separate good and bad payers
    oof_p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    oof_score = 100 * (1 - oof_p)
    bands = np.select([oof_score >= 90, oof_score >= 85, oof_score >= 80],
                      ["90+ GREAT", "85-90 OK", "80-85 NOT GREAT"], "<80 NOT ELIGIBLE")
    band_table = (pd.DataFrame({"band": bands, "defaulted": y})
                  .groupby("band")["defaulted"].agg(["size", "mean"]))
    order = ["90+ GREAT", "85-90 OK", "80-85 NOT GREAT", "<80 NOT ELIGIBLE"]
    metrics["default_rate_by_score_band"] = {
        b: {"applicants": int(band_table.loc[b, "size"]),
            "actual_default_rate": round(float(band_table.loc[b, "mean"]), 4)}
        for b in order if b in band_table.index
    }

    print("\n5-fold cross-validation (mean):")
    for k, v in metrics.items():
        if k != "default_rate_by_score_band":
            print(f"  {k:32s} {v}")
    print("\nActual default rate per score band (out-of-fold):")
    for b, r in metrics["default_rate_by_score_band"].items():
        print(f"  {b:18s} applicants={r['applicants']:5d}  default rate={r['actual_default_rate']:.1%}")

    model.fit(X, y)
    importances = (pd.Series(model.feature_importances_, index=X.columns)
                   .sort_values(ascending=False).head(15))
    metrics["top_features"] = {k: round(float(v), 4) for k, v in importances.items()}

    # Reference ranges from people who repaid, used to explain weak signals at scoring time
    repaid = X[y == 0]
    reference = {col: {"p10": round(float(repaid[col].quantile(0.10)), 4),
                       "p90": round(float(repaid[col].quantile(0.90)), 4)}
                 for col in ["savings_rate", "low_balance_share", "bill_payment_score",
                             "balance_to_income", "spend_to_inflow_ratio", "failed_txn_ratio",
                             "income_stability", "recharge_on_time_ratio"]}

    MODELS.mkdir(exist_ok=True)
    (MODELS / "reference_ranges.json").write_text(json.dumps(reference, indent=2))
    joblib.dump(model, MODELS / "model.joblib")
    (MODELS / "feature_columns.json").write_text(json.dumps(list(X.columns), indent=2))
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nTop features:")
    for k, v in importances.items():
        print(f"  {k:32s} {v:.4f}")
    print(f"\nSaved model and metrics to {MODELS}")


if __name__ == "__main__":
    main()
