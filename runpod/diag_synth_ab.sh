#!/bin/bash
# DIAG-AB-RUN-V1 — fetch + run the synth A/B diagnostic, detached-safe.
set -u
echo "DIAG-AB-RUN-V1 starting $(date -u)"
VENV=/workspace/gatefit-venv
LOG=/workspace/diag_ab.log
: > "$LOG"
( while :; do cp -f "$LOG" /workspace/ComfyUI/output/diag_ab_log_copy.txt 2>/dev/null; sleep 15; done ) &
MIR=$!
"$VENV/bin/python" -u /workspace/diag_synth_ab.py >> "$LOG" 2>&1
echo "DIAG-RUN-EXIT $?" >> "$LOG"
kill $MIR 2>/dev/null
cp -f "$LOG" /workspace/ComfyUI/output/diag_ab_log_copy.txt 2>/dev/null
echo "DIAG-AB-RUN-DONE"
