"""Tests for cost_ledger.py dry-run mode and per-project budgets."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from cost_ledger import BudgetBlocked, CostLedger


def ledger(tmp_path, **kw):
    return CostLedger(str(tmp_path / "spend.jsonl"), **kw)


def test_dry_run_never_raises_and_never_authorizes(tmp_path):
    l = ledger(tmp_path, dry_run=True)
    # wildly over cap: no raise, would_authorize=False
    r = l.authorize("eff-1", "draft", 999.00)
    assert r["dry_run"] is True
    assert r["authorized"] is False
    assert r["would_authorize"] is False
    assert r["block_reason"] is not None
    # under cap: would_authorize=True, still not authorized
    r2 = l.authorize("eff-2", "draft", 0.10)
    assert r2["would_authorize"] is True
    assert r2["authorized"] is False


def test_dry_run_actuals_do_not_count(tmp_path):
    l = ledger(tmp_path, dry_run=True)
    l.authorize("eff-1", "draft", 0.25)
    l.record_actual("eff-1", "draft", 9.99, "simulated bill")
    s = l.summary()
    assert s["day_spend_usd"] == 0.0
    assert s["month_spend_usd"] == 0.0
    assert s["dry_run"] is True
    # ...but the audit trail still has the lines (marked DRYRUN)
    lines = [json.loads(x) for x in
             open(str(tmp_path / "spend.jsonl")) if x.strip()]
    assert any("DRYRUN" in x["detail"] for x in lines)


def test_dry_run_hero_block_reported_not_raised(tmp_path):
    l = ledger(tmp_path, dry_run=True)
    r = l.authorize("eff-h", "hero", 5.00, hero_approved=False)
    assert r["would_authorize"] is False
    assert "hero" in r["block_reason"].lower()


def test_project_cap_blocks_per_project(tmp_path):
    l = ledger(tmp_path, project_caps={"trailer": 1.00})
    l.authorize("eff-1", "draft", 0.30, project="trailer")
    l.record_actual("eff-1", "draft", 0.90, project="trailer")
    with pytest.raises(BudgetBlocked, match="project 'trailer'"):
        l.authorize("eff-2", "draft", 0.20, project="trailer")
    # other projects / default bucket unaffected
    l.authorize("eff-3", "draft", 0.20, project="short")
    l.authorize("eff-4", "draft", 0.20)
    s = l.summary(project="trailer")
    assert s["project_spend_usd"] == 0.90
    assert s["project_cap_usd"] == 1.00


def test_project_without_cap_uses_global_caps_only(tmp_path):
    l = ledger(tmp_path, project_caps={"trailer": 1.00})
    l.authorize("eff-1", "final", 2.00, project="uncapped")
    l.record_actual("eff-1", "final", 2.00, project="uncapped")
    s = l.summary(project="uncapped")
    assert s["project_spend_usd"] == 2.00
    assert s["project_cap_usd"] is None


def test_legacy_entries_without_project_field_still_count(tmp_path):
    import time
    p = str(tmp_path / "spend.jsonl")
    with open(p, "w") as f:  # pre-change format: no "project" key
        f.write(json.dumps({"ts": time.time(), "kind": "actual",
                            "effect_id": "old", "tier": "draft",
                            "amount_usd": 1.50, "detail": ""}) + "\n")
    l = CostLedger(p, project_caps={"trailer": 2.00})
    # legacy entry counts toward the global month total...
    assert l.summary()["month_spend_usd"] == 1.50
    # ...but not toward any project bucket
    assert l.summary(project="trailer")["project_spend_usd"] == 0.0
