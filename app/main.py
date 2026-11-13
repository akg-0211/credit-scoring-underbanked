"""
FastAPI app: a screen to enter any custom applicant, plus a JSON API.

Run:  uvicorn app.main:app --reload
Open: http://localhost:8000        (screen)
      http://localhost:8000/docs   (API docs)
"""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.examples import EXAMPLES
from src.scoring import CreditScorer

app = FastAPI(title="Alternative Credit Scoring for Under-banked Individuals")
scorer = CreditScorer()
STATIC = Path(__file__).resolve().parent / "static"

Ratio = Field(ge=0, le=1)
Flag = Field(ge=0, le=1)


class Applicant(BaseModel):
    segment: Literal["gig_worker", "student", "thin_file", "informal_salaried"]
    age: int = Field(ge=18, le=80)
    city_tier: Literal[1, 2, 3]
    monthly_income: float = Field(ge=0)
    income_variability: float = Ratio
    employment_months: float = Field(ge=0)
    dependents: int = Field(ge=0)
    recharge_on_time_ratio: float = Ratio
    phone_bill_on_time_ratio: float = Ratio
    recharge_gaps_6m: int = Field(ge=0)
    avg_recharge_amount: float = Field(ge=0)
    sim_age_months: float = Field(ge=0)
    sim_changes_12m: int = Field(ge=0)
    utility_bill_on_time_ratio: float = Ratio
    rent_on_time_ratio: float = Ratio
    upi_monthly_inflow: float = Field(ge=0)
    upi_monthly_spend: float = Field(ge=0)
    upi_txn_count_month: int = Field(ge=0)
    pct_spend_essentials: float = Ratio
    pct_spend_entertainment: float = Ratio
    pct_spend_gambling: float = Ratio
    p2p_txn_ratio: float = Ratio
    failed_txn_ratio: float = Ratio
    late_night_txn_ratio: float = Ratio
    avg_balance: float = Field(ge=0)
    low_balance_days_month: int = Field(ge=0, le=31)
    num_gambling_apps: int = Field(ge=0)
    num_loan_apps: int = Field(ge=0)
    num_finance_apps: int = Field(ge=0)
    num_apps_total: int = Field(ge=0)
    device_type: Literal["phone", "tablet", "tv"]
    os_version_age_months: float = Field(ge=0)
    location_consistency_score: float = Ratio
    device_rooted: int = Flag
    vpn_active: int = Flag
    emulator_detected: int = Flag
    vm_detected: int = Flag
    frida_detected: int = Flag
    timezone_mismatch: int = Flag
    device_id_mismatch: int = Flag


@app.get("/")
def screen():
    return FileResponse(STATIC / "index.html")


@app.post("/api/score")
def score(applicant: Applicant):
    return scorer.score(applicant.model_dump())


@app.get("/api/examples")
def examples():
    return EXAMPLES


@app.get("/api/metrics")
def metrics():
    return scorer.metrics
