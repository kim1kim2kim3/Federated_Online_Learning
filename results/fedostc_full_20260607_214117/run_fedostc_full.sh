#!/usr/bin/env bash
set -uo pipefail
cd /home/y/Paper_2/Federated_Online_Learning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESULT_DIR="/home/y/Paper_2/Federated_Online_Learning/results/fedostc_full_20260607_214117"
echo $$ > "$RESULT_DIR/driver.pid"
echo running > "$RESULT_DIR/status"
{
  echo "[start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[cwd] $(pwd)"
  echo "[cmd] .venv/bin/python scripts/run_full_baseline_stream.py --agg_model fedostc --rounds 0 --progress_every 100 --cuda_memory_guard_mb 2048 --output_dir $RESULT_DIR"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
} | tee "$RESULT_DIR/master.log"
set +e
.venv/bin/python scripts/run_full_baseline_stream.py   --agg_model fedostc   --rounds 0   --progress_every 100   --cuda_memory_guard_mb 2048   --output_dir "$RESULT_DIR"   > "$RESULT_DIR/fedostc_stdout.log"   2> "$RESULT_DIR/fedostc_stderr.log"
CODE=$?
set -e
{
  echo "[finish] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[exit_code] $CODE"
  tail -n 5 "$RESULT_DIR/fedostc_progress.log" 2>/dev/null || true
} | tee -a "$RESULT_DIR/master.log"
echo "$CODE" > "$RESULT_DIR/exit_code"
if [ "$CODE" -eq 0 ]; then
  echo completed > "$RESULT_DIR/status"
else
  echo failed > "$RESULT_DIR/status"
fi
exit "$CODE"
