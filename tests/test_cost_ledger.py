"""Tests for tools/cost_ledger.py — cost caps enforced in code."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from cost_ledger import CostLedger, BudgetBlocked, EFFECT_CAPS


@pytest.fixture
def ledger(tmp_path):
    return CostLedger(str(tmp_path / "spend.jsonl"))


def test_authorize_draft_within_cap(ledger):
    auth = ledger.authorize("eff-1", "draft", 0.25)
    assert auth["authorized"] is True
    assert auth["estimate_usd"] == 0.25


def test_per_effect_cap_blocks(ledger):
    with pytest.raises(BudgetBlocked, match="per-effect cap"):
        ledger.authorize("eff-1", "draft", 0.31)  # over $0.30 draft cap
    with pytest.raises(BudgetBlocked, match="per-effect cap"):
        ledger.authorize("eff-1", "final", 2.01)  # over $2.00 final cap


def test_hero_requires_explicit_approval(ledger):
    with pytest.raises(BudgetBlocked, match="explicit approval"):
        ledger.authorize("eff-1", "hero", 5.00, hero_approved=False)
    auth = ledger.authorize("eff-1", "hero", 5.00, hero_approved=True)
    assert auth["authorized"] is True


def test_unknown_tier_rejected(ledger):
    with pytest.raises(ValueError, match="Unknown tier"):
        ledger.authorize("eff-1", "platinum", 0.10)


def test_estimate_logged_before_paid_call(ledger, tmp_path):
    # authorize() must write an estimate record BEFORE any actual spend.
    ledger.authorize("eff-9", "draft", 0.20)
    lines = open(tmp_path / "spend.jsonl").read().strip().split("\n")
    assert len(lines) == 1
    import json
    rec = json.loads(lines[0])
    assert rec["kind"] == "estimate" and rec["effect_id"] == "eff-9"


def test_daily_hard_cap_blocks(ledger):
    # Spend $9.90 of the $10 daily cap, then a $0.20 estimate must block.
    for i in range(33):
        ledger.authorize(f"eff-d{i}", "draft", 0.30)
        ledger.record_actual(f"eff-d{i}", "draft", 0.30)
    with pytest.raises(BudgetBlocked, match="Daily spend cap"):
        ledger.authorize("eff-over", "draft", 0.20)


def test_daily_soft_cap_alerts_but_allows(ledger, tmp_path):
    ledger.authorize("eff-a", "final", 2.00)
    ledger.record_actual("eff-a", "final", 4.90)
    auth = ledger.authorize("eff-b", "draft", 0.30)  # crosses $5 soft cap
    assert auth["authorized"] is True
    import json
    kinds = [json.loads(l)["kind"] for l in open(tmp_path / "spend.jsonl")]
    assert "alert" in kinds


def test_monthly_cap_blocks(ledger, tmp_path):
    # Seed $100 of September actuals directly (2 x $2.00/day x 25 days),
    # then a new $2.00 estimate must trip the monthly hard cap.
    import json as _json
    from datetime import datetime, timezone
    path = tmp_path / "spend.jsonl"
    with open(path, "a") as f:
        for d in range(1, 26):
            ts = datetime(2026, 9, d, 12, tzinfo=timezone.utc).timestamp()
            for j in range(2):
                f.write(_json.dumps({"ts": ts, "kind": "actual",
                                      "effect_id": f"eff-s{d}-{j}",
                                      "tier": "final", "amount_usd": 2.00,
                                      "detail": ""}) + "\n")
    assert ledger.summary()["month_spend_usd"] == 0.0  # today is October
    # Re-point "now" into September via a subclass-free trick: authorize uses
    # _month_key(); patch it to September so the seeded spend counts.
    ledger._month_key = lambda ts=None: "2026-09"  # noqa: SLF001
    with pytest.raises(BudgetBlocked, match="Monthly spend cap"):
        ledger.authorize("eff-over", "final", 2.00)


def test_failed_generations_record_zero(ledger):
    ledger.authorize("eff-f", "draft", 0.25)
    ledger.record_actual("eff-f", "draft", 0.00, "failed generation, not billed")
    s = ledger.summary()
    assert s["day_spend_usd"] == 0.0


def test_summary_shape(ledger):
    s = ledger.summary()
    assert s["effect_caps"] == EFFECT_CAPS
    assert s["daily_hard_cap"] == 10.00
    assert s["monthly_hard_cap"] == 100.00
