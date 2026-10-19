#!/bin/bash
# EXPORT-SCAFFOLD-V1 — rerun the V8 deep fit and SAVE the body mesh.
set -x
exec 9>/workspace/gate_test.lock
if ! flock -n 9; then echo "EXPORT-ALREADY-RUNNING"; exit 0; fi
VENV=/workspace/gatefit-venv
INDIR=/workspace/ComfyUI/input/gate_candidates
WORK=/workspace/gate_test
[ -x "$VENV/bin/python" ] || { echo "EXPORT-FAIL: gatefit venv missing"; exit 1; }
cp -f /workspace/export_scaffold.py "$WORK/" 2>/dev/null || true
( while :; do cp -f /workspace/export_scaffold.log /workspace/ComfyUI/output/export_scaffold_log_copy.txt 2>/dev/null; sleep 20; done ) &
MPID=$!
trap 'kill $MPID 2>/dev/null' EXIT
"$VENV/bin/python" -u /workspace/export_scaffold.py "$INDIR" "$WORK" 2>&1 | tee -a /workspace/export_scaffold.log
cp -f "$WORK/scaffold_body.ply" /workspace/ComfyUI/output/scaffold_body.ply && echo "PLY-COPIED-TO-OUTPUT"
echo "EXPORT-SCAFFOLD-SCRIPT-DONE"
