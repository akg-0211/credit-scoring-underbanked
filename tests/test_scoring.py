from src.examples import EXAMPLES
from src.scoring import CreditScorer

scorer = CreditScorer()


def test_good_payer_is_white_and_eligible():
    r = scorer.score(EXAMPLES["good"])
    assert r["category"] == "WHITE"
    assert r["eligible"] is True
    assert r["score"] >= 90


def test_any_red_flag_rejects_outright():
    for flag in ["device_rooted", "vpn_active", "emulator_detected", "vm_detected",
                 "frida_detected", "timezone_mismatch", "device_id_mismatch"]:
        r = scorer.score({**EXAMPLES["good"], flag: 1})
        assert r["category"] == "RED"
        assert r["score"] == 0
        assert r["eligible"] is False


def test_tv_device_is_red():
    assert scorer.score({**EXAMPLES["good"], "device_type": "tv"})["category"] == "RED"


def test_grey_flags_reduce_score():
    good = scorer.score(EXAMPLES["good"])
    grey = scorer.score(EXAMPLES["grey"])
    assert grey["category"] == "GREY"
    assert grey["score"] < good["score"]
    assert grey["model"]["grey_penalty"] > 0


def test_decision_thresholds():
    assert CreditScorer.decision_for(95) == "GREAT"
    assert CreditScorer.decision_for(87) == "OK"
    assert CreditScorer.decision_for(82) == "NOT GREAT"
    assert CreditScorer.decision_for(79.9) == "NOT ELIGIBLE"
