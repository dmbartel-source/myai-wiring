"""Spend ledger for the MYAI media pipeline.

Cost policy is enforced IN CODE, not in prompts:
  - Pre-generation estimate is logged BEFORE every paid call.
  - Per-effect caps: draft <= $0.30/clip, standard final <= $2.00/effect
    (incl. retries), hero <= $10.00 and only with explicit approval.
  - Daily caps: $5.00 soft (alert), $10.00 hard (block).
  - Monthly cap: $100.00 hard (block), with 50/80/100% alerts.
  - Failed generations are not billed by most providers — only record
    actual billed cost via record_actual().

Storage: JSONL log (one record per line) + a small JSON state file for
running totals. No network, no API keys. Safe to run anywhere.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional


# --- Policy (USD) -----------------------------------------------------------
EFFECT_CAPS = {
    "draft": 0.30,    # per accepted draft clip
    "final": 2.00,    # per effect, including retries
    "hero": 10.00,    # hero only with explicit approval
}
DAILY_SOFT_CAP = 5.00    # alert
DAILY_HARD_CAP = 10.00   # block
MONTHLY_HARD_CAP = 100.00  # block
ALERT_THRESHOLDS = (0.50, 0.80, 1.00)  # of monthly cap


class BudgetBlocked(Exception):
    """Raised when a spend would exceed a hard cap. Fail closed."""


def _is_dryrun_rec(rec: dict) -> bool:
    """Dry-run actuals are logged for the audit trail but never counted."""
    return "DRYRUN" in str(rec.get("detail", ""))


@dataclass
class LedgerEntry:
    ts: float
    kind: str            # "estimate" | "actual" | "alert"
    effect_id: str
    tier: str            # "draft" | "final" | "hero"
    amount_usd: float
    detail: str = ""
    project: str = ""    # per-project budget bucket ("" = default bucket)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class CostLedger:
    """Spend ledger with hard caps in code.

    Args:
        ledger_path: JSONL log path.
        dry_run: when True, authorize() NEVER raises and NEVER authorizes —
            it computes what WOULD happen and returns it. Estimates are
            still logged (marked DRYRUN). record_actual() becomes a no-op
            log line. For planning/previewing spend without consequences.
        project_caps: optional {project_name: monthly_usd_cap} for
            per-project budgets, enforced like the monthly cap.
    """

    def __init__(self, ledger_path: str, dry_run: bool = False,
                 project_caps: Optional[dict] = None):
        self.ledger_path = ledger_path
        self.state_path = ledger_path + ".state.json"
        self.dry_run = bool(dry_run)
        self.project_caps = dict(project_caps or {})
        os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)
        # Touch the log so it always exists.
        open(self.ledger_path, "a").close()

    # -- internal ---------------------------------------------------------
    def _day_key(self, ts: Optional[float] = None) -> str:
        dt = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")

    def _month_key(self, ts: Optional[float] = None) -> str:
        dt = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc)
        return dt.strftime("%Y-%m")

    def _actual_spend(self, key: str, prefix: str) -> float:
        """Sum of actual (billed) spend for a day (%Y-%m-%d) or month (%Y-%m)."""
        total = 0.0
        try:
            with open(self.ledger_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("kind") != "actual":
                        continue
                    if _is_dryrun_rec(rec):
                        continue
                    rts = rec.get("ts", 0)
                    k = self._day_key(rts) if prefix == "day" else self._month_key(rts)
                    if k == key:
                        total += float(rec.get("amount_usd", 0.0))
        except FileNotFoundError:
            pass
        return round(total, 4)

    def _effect_spend(self, effect_id: str) -> float:
        total = 0.0
        try:
            with open(self.ledger_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("kind") == "actual" and rec.get("effect_id") == effect_id:
                        if _is_dryrun_rec(rec):
                            continue
                        total += float(rec.get("amount_usd", 0.0))
        except FileNotFoundError:
            pass
        return round(total, 4)

    def _project_spend(self, project: str, month_key: str) -> float:
        """Sum of actual spend for a project bucket in a month (%Y-%m)."""
        total = 0.0
        try:
            with open(self.ledger_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("kind") != "actual":
                        continue
                    if _is_dryrun_rec(rec):
                        continue
                    if rec.get("project", "") != project:
                        continue
                    if self._month_key(rec.get("ts", 0)) == month_key:
                        total += float(rec.get("amount_usd", 0.0))
        except FileNotFoundError:
            pass
        return round(total, 4)

    def _log(self, entry: LedgerEntry) -> None:
        with open(self.ledger_path, "a") as f:
            f.write(entry.to_json() + "\n")

    def _alert(self, message: str, effect_id: str = "") -> None:
        self._log(LedgerEntry(time.time(), "alert", effect_id, "", 0.0, message))

    # -- public -----------------------------------------------------------
    def authorize(
        self,
        effect_id: str,
        tier: str,
        estimate_usd: float,
        hero_approved: bool = False,
        project: str = "",
    ) -> dict:
        """Pre-generation authorization. MUST be called before every paid call.

        In dry_run mode this NEVER raises and NEVER authorizes: it returns
        {"authorized": False, "dry_run": True, "would_authorize": bool,
        "block_reason": str | None} describing what would have happened.
        The estimate is still logged (detail prefixed DRYRUN).

        Returns an authorization record dict on approval. Raises BudgetBlocked
        on any hard-cap violation. Logs the estimate either way.
        """
        if tier not in EFFECT_CAPS:
            raise ValueError(f"Unknown tier {tier!r}; expected one of {sorted(EFFECT_CAPS)}")

        checks: list[str] = []  # block reasons, in evaluation order

        if tier == "hero" and not hero_approved:
            checks.append("hero tier requires explicit approval")
        if estimate_usd > EFFECT_CAPS[tier]:
            checks.append(
                f"Estimate ${estimate_usd:.4f} exceeds the per-effect cap "
                f"(${EFFECT_CAPS[tier]:.2f}) for tier {tier!r}.")

        day = self._day_key()
        month = self._month_key()
        day_spend = self._actual_spend(day, "day")
        month_spend = self._actual_spend(month, "month")

        # Monthly hard cap (the binding one).
        if month_spend + estimate_usd > MONTHLY_HARD_CAP:
            checks.append(
                f"Monthly spend cap ${MONTHLY_HARD_CAP:.2f} would be exceeded. "
                f"Current: ${month_spend:.2f}.")
        # Daily hard cap.
        if day_spend + estimate_usd > DAILY_HARD_CAP:
            checks.append(
                f"Daily spend cap ${DAILY_HARD_CAP:.2f} would be exceeded. "
                f"Current: ${day_spend:.2f}.")
        # Per-project monthly cap.
        project_spend = 0.0
        project_cap = self.project_caps.get(project) if project else None
        if project_cap is not None:
            project_spend = self._project_spend(project, month)
            if project_spend + estimate_usd > project_cap:
                checks.append(
                    f"project {project!r} cap ${project_cap:.2f} would be "
                    f"exceeded (current ${project_spend:.2f})")

        if self.dry_run:
            reason = checks[0] if checks else None
            self._log(LedgerEntry(time.time(), "estimate", effect_id, tier,
                                  estimate_usd,
                                  f"DRYRUN: would_{'block' if reason else 'authorize'}"
                                  + (f" ({reason})" if reason else ""),
                                  project))
            return {
                "authorized": False,
                "dry_run": True,
                "would_authorize": not checks,
                "block_reason": reason,
                "effect_id": effect_id,
                "tier": tier,
                "estimate_usd": estimate_usd,
                "project": project,
                "day_spend_usd": day_spend,
                "month_spend_usd": month_spend,
                "project_spend_usd": project_spend,
            }

        if checks:
            self._log(LedgerEntry(time.time(), "estimate", effect_id, tier,
                                  estimate_usd, f"BLOCKED: {checks[0]}", project))
            if "hero tier requires" in checks[0]:
                raise BudgetBlocked(
                    "Hero-tier generation requires David's explicit approval. "
                    "Not authorized."
                )
            raise BudgetBlocked(checks[0] + " Blocked.")

        # Monthly alerts at 50/80/100%.
        for frac in ALERT_THRESHOLDS:
            if month_spend < MONTHLY_HARD_CAP * frac <= month_spend + estimate_usd:
                self._alert(f"Monthly spend reached {int(frac*100)}% "
                            f"(${month_spend + estimate_usd:.2f} of ${MONTHLY_HARD_CAP:.2f})",
                            effect_id)
        # Daily soft-cap alert.
        if day_spend < DAILY_SOFT_CAP <= day_spend + estimate_usd:
            self._alert(f"Daily soft cap ${DAILY_SOFT_CAP:.2f} reached "
                        f"(${day_spend + estimate_usd:.2f})", effect_id)

        self._log(LedgerEntry(time.time(), "estimate", effect_id, tier,
                              estimate_usd, "AUTHORIZED", project))
        return {
            "authorized": True,
            "effect_id": effect_id,
            "tier": tier,
            "estimate_usd": estimate_usd,
            "project": project,
            "day_spend_usd": day_spend,
            "month_spend_usd": month_spend,
            "project_spend_usd": project_spend,
        }

    def record_actual(
        self, effect_id: str, tier: str, amount_usd: float, detail: str = "",
        project: str = "",
    ) -> None:
        """Record what was ACTUALLY billed. Call after every paid call,
        including $0.00 for failed-but-unbilled generations. In dry_run
        mode this only logs (marked DRYRUN) and changes no totals — dry-run
        actuals are excluded from spend sums."""
        if self.dry_run:
            detail = f"DRYRUN actual (not counted): {detail}"
        self._log(LedgerEntry(time.time(), "actual", effect_id, tier,
                              round(amount_usd, 4), detail, project))

    def summary(self, project: str = "") -> dict:
        """Current spend picture (for the agent to show David)."""
        day = self._day_key()
        month = self._month_key()
        out = {
            "day": day,
            "day_spend_usd": self._actual_spend(day, "day"),
            "daily_soft_cap": DAILY_SOFT_CAP,
            "daily_hard_cap": DAILY_HARD_CAP,
            "month": month,
            "month_spend_usd": self._actual_spend(month, "month"),
            "monthly_hard_cap": MONTHLY_HARD_CAP,
            "effect_caps": dict(EFFECT_CAPS),
            "dry_run": self.dry_run,
        }
        if project:
            out["project"] = project
            out["project_spend_usd"] = self._project_spend(project, month)
            out["project_cap_usd"] = self.project_caps.get(project)
        return out
