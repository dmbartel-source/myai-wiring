"""Plan-on-disk pipeline executor for the MYAI media pipeline.

Takes an EDL (Edit Decision List) JSON plan and runs deterministic stages
with checkpoints between phases. Resumable: re-running an EDL picks up
after the last completed checkpoint.

Design rules (from the media-agent research):
  - LLM plans; THIS code executes exactly what it is told.
  - Nothing lives only in LLM context: plan, state, and manifests on disk.
  - File/SQLite job store — no message broker.
  - At-least-once + idempotent steps: every stage declares an idempotency
    key; completed stages are never re-run.
  - Failure ladder: retry transient (jittered backoff, strict timeouts) ->
    ordered provider chain -> degrade to cheaper model -> loud failure with
    a fallback trail. Validation failures fail fast (0 retries).

EDL schema (validated by validate_edl):
  {
    "edl_version": 1,
    "job_id": "effect-20261008-001",
    "stages": [
      {"id": "gen-draft", "tool": "fal_submit",
       "args": {"route": "wan-25", "prompt": "...", "duration_s": 5},
       "idempotency_key": "myai-abc123", "checkpoint": true},
      ...
    ]
  }

Built-in stage tools: fal_submit, fal_wait, download, ffmpeg, shell,
checkpoint (no-op marker), qa (video_qa tiers 0-1), local_fx (deterministic
local ffmpeg recipes — $0, fully private), local_gen (self-hosted diffusion
hook; raises LocalGenNotWired until wired). Custom tools can be
registered via register_tool(name, fn).

Policy gates (fail closed, before any network or spend):
  - fal_submit: ToS gate (fal_client.require_route refuses rejected models
    outright; conditional models need tos_ack=True) + copyright-privacy
    gate (privacy.authorize_third_party needs explicit per-call
    allow_third_party=True; DeepSeek ban enforced in code).
  - download: the egress gate (tools/net.py) allowlists artifact hosts and
    logs every attempt; anything else raises EgressBlocked.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3
import subprocess
import time
from typing import Any, Callable, Optional


EDL_VERSION = 1
JOB_LAYOUT = ("inputs", "takes", "work", "qa", "final", "checkpoints")

# Policy-refusal exceptions: ToS rejections, privacy blocks, egress blocks,
# and the unwired-local-gen stub are NEVER transient — the failure ladder
# must fail fast on them instead of burning retry budgets. Imported
# defensively for both package and script-dir layouts.
try:
    from .fal_client import FalTosError
    from .privacy import PrivacyBlocked
    from .net import EgressBlocked
    from .local_gen import LocalGenNotWired
    _POLICY_REFUSALS = (FalTosError, PrivacyBlocked, EgressBlocked,
                        LocalGenNotWired)
except ImportError:
    try:
        from fal_client import FalTosError
        from privacy import PrivacyBlocked
        from net import EgressBlocked
        from local_gen import LocalGenNotWired
        _POLICY_REFUSALS = (FalTosError, PrivacyBlocked, EgressBlocked,
                            LocalGenNotWired)
    except ImportError:
        _POLICY_REFUSALS = ()


class PipelineError(Exception):
    pass


class EDLValidationError(PipelineError):
    pass


class StageFailed(PipelineError):
    def __init__(self, stage_id: str, reason: str, trail: Optional[list] = None):
        super().__init__(f"Stage {stage_id!r} failed: {reason}")
        self.stage_id = stage_id
        self.trail = trail or []


def validate_edl(edl: dict) -> dict:
    """Validate an EDL plan. Returns it unchanged or raises EDLValidationError."""
    if not isinstance(edl, dict):
        raise EDLValidationError("EDL must be a JSON object")
    if edl.get("edl_version") != EDL_VERSION:
        raise EDLValidationError(
            f"Unsupported edl_version {edl.get('edl_version')!r}; expected {EDL_VERSION}")
    if not edl.get("job_id") or not isinstance(edl["job_id"], str):
        raise EDLValidationError("EDL needs a string job_id")
    stages = edl.get("stages")
    if not isinstance(stages, list) or not stages:
        raise EDLValidationError("EDL needs a non-empty stages list")
    seen_ids, seen_keys = set(), set()
    for i, st in enumerate(stages):
        if not isinstance(st, dict):
            raise EDLValidationError(f"Stage {i} must be an object")
        for field in ("id", "tool", "args", "idempotency_key"):
            if field not in st:
                raise EDLValidationError(f"Stage {i} missing required field {field!r}")
        if st["id"] in seen_ids:
            raise EDLValidationError(f"Duplicate stage id {st['id']!r}")
        if st["idempotency_key"] in seen_keys:
            raise EDLValidationError(
                f"Duplicate idempotency_key {st['idempotency_key']!r}")
        seen_ids.add(st["id"])
        seen_keys.add(st["idempotency_key"])
        if not isinstance(st["args"], dict):
            raise EDLValidationError(f"Stage {st['id']!r}: args must be an object")
    return edl


class JobStore:
    """SQLite job store: pending -> running -> done | failed. One row per stage."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS stage_state (
                       job_id TEXT, stage_id TEXT, idempotency_key TEXT,
                       status TEXT, result_json TEXT, updated_ts REAL,
                       PRIMARY KEY (job_id, stage_id))"""
            )

    def get(self, job_id: str, stage_id: str) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as c:
            row = c.execute(
                "SELECT status, result_json FROM stage_state WHERE job_id=? AND stage_id=?",
                (job_id, stage_id)).fetchone()
        if row:
            return {"status": row[0],
                    "result": json.loads(row[1]) if row[1] else None}
        return None

    def set(self, job_id: str, stage_id: str, idem_key: str,
            status: str, result: Optional[dict] = None) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                """INSERT INTO stage_state
                   (job_id, stage_id, idempotency_key, status, result_json, updated_ts)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(job_id, stage_id) DO UPDATE SET
                     status=excluded.status, result_json=excluded.result_json,
                     updated_ts=excluded.updated_ts""",
                (job_id, stage_id, idem_key, status,
                 json.dumps(result) if result is not None else None, time.time()))

    def job_summary(self, job_id: str) -> list[dict]:
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute(
                "SELECT stage_id, status FROM stage_state WHERE job_id=? ORDER BY updated_ts",
                (job_id,)).fetchall()
        return [{"stage_id": r[0], "status": r[1]} for r in rows]


def _jittered_backoff(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    return min(cap, base ** attempt + random.uniform(0, 1.0))


class Pipeline:
    """Executes an EDL with checkpoints, resume, and a failure ladder."""

    def __init__(self, jobs_root: str, max_retries: int = 3):
        self.jobs_root = jobs_root
        self.store = JobStore(os.path.join(jobs_root, "jobs.sqlite"))
        self.tools: dict[str, Callable[[dict, dict], dict]] = {}
        self.max_retries = max_retries
        self._register_builtin_tools()

    # -- job layout ------------------------------------------------------
    def job_dir(self, job_id: str) -> str:
        d = os.path.join(self.jobs_root, job_id)
        os.makedirs(d, exist_ok=True)
        for sub in JOB_LAYOUT:
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        return d

    def register_tool(self, name: str,
                      fn: Callable[[dict, dict], dict]) -> None:
        """Register a stage tool. fn(args, ctx) -> result dict.
        ctx carries: job_id, job_dir, stage_id, attempt.
        Raise StageFailed on failure. Tools must be idempotent."""
        self.tools[name] = fn

    # -- built-in tools ---------------------------------------------------
    def _register_builtin_tools(self) -> None:
        self.register_tool("checkpoint", lambda args, ctx: {"marked": True})
        self.register_tool("shell", self._tool_shell)
        self.register_tool("ffmpeg", self._tool_ffmpeg)
        self.register_tool("qa", self._tool_qa)
        self.register_tool("fal_submit", self._tool_fal_submit)
        self.register_tool("fal_wait", self._tool_fal_wait)
        self.register_tool("download", self._tool_download)
        self.register_tool("local_fx", self._tool_local_fx)
        self.register_tool("local_gen", self._tool_local_gen)

    def _tool_shell(self, args: dict, ctx: dict) -> dict:
        cmd = args.get("cmd")
        if not cmd or not isinstance(cmd, list):
            raise StageFailed(ctx["stage_id"], "shell needs args.cmd as a list")
        timeout = int(args.get("timeout_s", 300))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=ctx["job_dir"])
        except subprocess.TimeoutExpired:
            raise StageFailed(ctx["stage_id"], f"shell timed out after {timeout}s")
        if proc.returncode != 0:
            raise StageFailed(ctx["stage_id"],
                              f"exit {proc.returncode}: {proc.stderr[:300]}")
        return {"stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}

    def _tool_ffmpeg(self, args: dict, ctx: dict) -> dict:
        if shutil.which("ffmpeg") is None:
            raise StageFailed(ctx["stage_id"], "ffmpeg not installed")
        cmd = ["ffmpeg", "-hide_banner", "-y"] + args.get("argv", [])
        if not args.get("argv"):
            raise StageFailed(ctx["stage_id"], "ffmpeg needs args.argv list")
        return self._tool_shell({"cmd": cmd,
                                 "timeout_s": args.get("timeout_s", 900)}, ctx)

    def _tool_qa(self, args: dict, ctx: dict) -> dict:
        try:
            from . import video_qa
        except ImportError:  # tools/ used as plain scripts dir
            import video_qa
        path = args["path"]
        report = video_qa.run_all(
            path, reference=args.get("reference"),
            expect=args.get("expect"),
            decision_points=args.get("decision_points"),
            frames_dir=os.path.join(ctx["job_dir"], "qa",
                                    ctx["stage_id"] + "_frames"))
        if not report["overall_pass"]:
            failed = [v["tier"] for v in report["verdicts"]
                      if not v["skipped"] and not v["pass"]]
            raise StageFailed(ctx["stage_id"], f"QC failed tiers: {failed}",
                              trail=[report])
        return {"report": report}

    def _tool_fal_submit(self, args: dict, ctx: dict) -> dict:
        try:
            from . import fal_client, cost_ledger, privacy
        except ImportError:  # tools/ used as plain scripts dir
            import fal_client, cost_ledger, privacy
        route = args["route"]
        tos_ack = bool(args.get("tos_ack", False))
        # ToS gate (refuses rejected models outright; conditional needs ack)
        # and copyright-privacy gate (explicit per-call third-party consent,
        # DeepSeek ban enforced in code). Both fail closed, before any
        # estimate is even computed.
        spec = fal_client.require_route(route, tos_ack=tos_ack)
        asset = privacy.Asset(
            path=args.get("source_media", ""),
            sensitivity=args.get("sensitivity", "sensitive"),
        )
        privacy.authorize_third_party(
            asset,
            allow_third_party=bool(args.get("allow_third_party", False)),
            route_tos="conditional+ack" if spec["tos"] == "conditional" else spec["tos"],
            brain=args.get("brain") or os.environ.get("MYAI_MEDIA_BRAIN"),
        )
        ledger = cost_ledger.CostLedger(
            os.path.join(ctx["job_dir"], "work", "spend.jsonl"),
            dry_run=bool(args.get("dry_run", False)),
            project_caps=args.get("project_caps"),
        )
        duration_s = float(args["duration_s"])
        estimate = fal_client.estimate_cost_usd(route, duration_s, tos_ack=tos_ack)
        tier = fal_client.MODEL_ROUTES[route]["tier"]
        auth = ledger.authorize(ctx["job_id"], tier, estimate,
                                hero_approved=bool(args.get("hero_approved")),
                                project=args.get("project", ""))
        job = fal_client.submit(route, args["arguments"],
                                webhook_url=args.get("webhook_url"),
                                idempotency_key=args.get("idempotency_key"),
                                tos_ack=tos_ack)
        ledger.record_actual(ctx["job_id"], tier, 0.0,
                             f"submitted {job.request_id} (billed on completion)",
                             project=args.get("project", ""))
        return {"request_id": job.request_id, "estimate_usd": estimate,
                "auth": auth}

    def _tool_fal_wait(self, args: dict, ctx: dict) -> dict:
        try:
            from . import fal_client, cost_ledger
        except ImportError:  # tools/ used as plain scripts dir
            import fal_client, cost_ledger
        try:
            from .fal_client import FalJob, JobStatus
        except ImportError:
            from fal_client import FalJob, JobStatus
        job = FalJob(request_id=args["request_id"],
                     endpoint=args["endpoint"],
                     idempotency_key=args.get("idempotency_key", ""))
        result = fal_client.wait(job, timeout_s=float(args.get("timeout_s", 600)))
        dest = os.path.join(ctx["job_dir"], "takes", args.get("filename", "take.mp4"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        video_url = _first_video_url(result)
        if not video_url:
            raise StageFailed(ctx["stage_id"], "completed job had no video URL")
        fal_client.download_to_file(video_url, dest)
        # Record actual billed cost if the provider reported it.
        billed = result.get("billed_usd")
        if billed is not None:
            ledger = cost_ledger.CostLedger(
                os.path.join(ctx["job_dir"], "work", "spend.jsonl"))
            ledger.record_actual(ctx["job_id"], args.get("tier", "final"),
                                 float(billed), f"job {job.request_id}")
        return {"file": dest, "result": result}

    def _tool_download(self, args: dict, ctx: dict) -> dict:
        from . import fal_client
        dest = os.path.join(ctx["job_dir"], "takes", args["filename"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # The URL goes through the egress gate inside download_to_file:
        # only allowlisted artifact hosts are fetched, everything else
        # raises EgressBlocked (and is logged).
        fal_client.download_to_file(args["url"], dest)
        return {"file": dest}

    def _tool_local_fx(self, args: dict, ctx: dict) -> dict:
        """Deterministic local effect. $0, fully private, no approval gate
        needed for spend (there is none)."""
        try:
            from . import local_effects
        except ImportError:  # tools/ used as plain scripts dir
            import local_effects
        recipe = args["recipe"]
        params = args.get("params", {})
        dest = os.path.join(ctx["job_dir"], "takes", args["filename"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        src = args.get("input")
        if not src:
            raise StageFailed(ctx["stage_id"], "local_fx needs args.input")
        spec = local_effects.RECIPES.get(recipe, {})
        if spec.get("two_inputs"):
            src2 = args.get("input2")
            if not src2:
                raise StageFailed(ctx["stage_id"],
                                  f"local_fx {recipe!r} needs args.input2")
            local_effects.apply2(src, src2, dest, recipe, params)
        else:
            local_effects.apply(src, dest, recipe, params)
        return {"file": dest, "recipe": recipe, "params": params,
                "cost_usd": 0.0, "private": True}

    def _tool_local_gen(self, args: dict, ctx: dict) -> dict:
        """Self-hosted diffusion hook. Raises LocalGenNotWired (loud) until
        the endpoint is configured and enabled — see tools/local_gen.py."""
        try:
            from . import local_gen
        except ImportError:  # tools/ used as plain scripts dir
            import local_gen
        cfg = local_gen.LocalGenConfig.from_env()
        result = local_gen.generate(args.get("prompt", ""), cfg,
                                    **args.get("gen_args", {}))
        return result

    # -- execution ---------------------------------------------------------
    def run(self, edl: dict) -> dict:
        """Run an EDL to completion. Resumes from the last checkpoint on
        re-run. Returns a manifest dict. Raises StageFailed (loud) on
        unrecoverable failure — never silently drops a stage."""
        edl = validate_edl(edl)
        job_id = edl["job_id"]
        job_dir = self.job_dir(job_id)
        manifest_path = os.path.join(job_dir, "manifest.json")
        manifest = self._load_manifest(manifest_path, edl)
        trail: list[str] = []

        for stage in edl["stages"]:
            sid = stage["id"]
            existing = self.store.get(job_id, sid)
            if existing and existing["status"] == "done":
                trail.append(f"{sid}: skipped (already done, idempotent)")
                continue  # idempotent resume
            tool = self.tools.get(stage["tool"])
            if tool is None:
                raise StageFailed(sid, f"unknown tool {stage['tool']!r}")
            ctx = {"job_id": job_id, "job_dir": job_dir,
                   "stage_id": sid, "attempt": 0}
            result = self._run_stage_with_ladder(stage, tool, ctx)
            self.store.set(job_id, sid, stage["idempotency_key"], "done", result)
            manifest["stages"][sid] = {"status": "done", "result": result}
            if stage.get("checkpoint"):
                self._write_checkpoint(job_dir, sid, manifest)
            trail.append(f"{sid}: done")

        manifest["status"] = "done"
        manifest["trail"] = trail
        self._save_manifest(manifest_path, manifest)
        return manifest

    def _run_stage_with_ladder(self, stage: dict, tool: Callable,
                               ctx: dict) -> dict:
        """Failure ladder: transient retries -> degrade -> loud failure."""
        sid = stage["id"]
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            ctx["attempt"] = attempt
            self.store.set(ctx["job_id"], sid, stage["idempotency_key"],
                           "running")
            try:
                return tool(stage["args"], ctx)
            except _POLICY_REFUSALS as e:
                # Policy refusals are never transient: fail fast and loud,
                # no retries, no fallback shopping.
                raise StageFailed(sid, f"policy refusal, not retried: {e}",
                                  trail=[f"{type(e).__name__}: {e}"])
            except StageFailed as e:
                last_err = e
                # Fail fast on validation-class errors (0 retries).
                msg = str(e).lower()
                if any(k in msg for k in ("validation", "invalid", "unknown tool",
                                          "not installed", "missing")):
                    raise
                if attempt < self.max_retries:
                    time.sleep(_jittered_backoff(attempt))
            except Exception as e:  # unexpected: treat as transient once
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(_jittered_backoff(attempt))
        # Degrade step: if the stage declared a cheaper fallback, try it once.
        fallback = stage.get("fallback")
        if fallback:
            try:
                ftool = self.tools.get(fallback["tool"])
                if ftool:
                    return ftool(fallback.get("args", {}), ctx)
            except Exception as e:
                last_err = e
        # Loud failure with the full trail — never silent.
        raise StageFailed(sid, f"exhausted retries: {last_err}",
                          trail=[str(last_err)])

    # -- manifests / checkpoints -------------------------------------------
    def _load_manifest(self, path: str, edl: dict) -> dict:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {"job_id": edl["job_id"], "edl_version": EDL_VERSION,
                "status": "running", "stages": {}}

    def _save_manifest(self, path: str, manifest: dict) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, path)  # atomic

    def _write_checkpoint(self, job_dir: str, stage_id: str, manifest: dict) -> None:
        cp = os.path.join(job_dir, "checkpoints", f"{stage_id}.json")
        tmp = cp + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"stage_id": stage_id, "ts": time.time(),
                       "manifest": manifest}, f, indent=2)
        os.replace(tmp, cp)


def _first_video_url(result: dict) -> Optional[str]:
    """Pull the first video URL out of a fal.ai result payload."""
    def walk(o: Any) -> Optional[str]:
        if isinstance(o, dict):
            url = o.get("url", "")
            if isinstance(url, str) and url.startswith("http"):
                ct = str(o.get("content_type", ""))
                if "video" in ct or url.endswith((".mp4", ".webm", ".mov")):
                    return url
            for v in o.values():
                found = walk(v)
                if found:
                    return found
        elif isinstance(o, list):
            for v in o:
                found = walk(v)
                if found:
                    return found
        return None
    found = walk(result.get("video") or result)
    return found
