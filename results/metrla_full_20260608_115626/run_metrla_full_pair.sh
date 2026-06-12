#!/usr/bin/env bash
set -uo pipefail
cd /home/y/Paper_2/Federated_Online_Learning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESULT_DIR="/home/y/Paper_2/Federated_Online_Learning/results/metrla_full_20260608_115626"
CONFIG_PATH="/home/y/Paper_2/Federated_Online_Learning/results/metrla_full_20260608_115626/metr_la_config.yaml"
echo $$ > "$RESULT_DIR/driver.pid"
echo running_refol > "$RESULT_DIR/status"
{
  echo "[start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[cwd] $(pwd)"
  echo "[config] $CONFIG_PATH"
  echo "[refol_cmd] .venv/bin/python scripts/run_full_baseline_stream.py --agg_model refol --config_path $CONFIG_PATH --rounds 0 --progress_every 100 --cuda_memory_guard_mb 4096 --output_dir $RESULT_DIR"
  echo "[fedostc_cmd] .venv/bin/python scripts/run_full_baseline_stream.py --agg_model fedostc --config_path $CONFIG_PATH --rounds 0 --progress_every 100 --cuda_memory_guard_mb 4096 --output_dir $RESULT_DIR"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
} | tee "$RESULT_DIR/master.log"

run_one() {
  local agg="$1"
  echo running_"$agg" > "$RESULT_DIR/status"
  echo "[$agg start] $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$RESULT_DIR/master.log"
  set +e
  .venv/bin/python scripts/run_full_baseline_stream.py     --agg_model "$agg"     --config_path "$CONFIG_PATH"     --rounds 0     --progress_every 100     --cuda_memory_guard_mb 4096     --output_dir "$RESULT_DIR"     > "$RESULT_DIR/${agg}_stdout.log"     2> "$RESULT_DIR/${agg}_stderr.log"
  local code=$?
  set -e
  echo "$code" > "$RESULT_DIR/${agg}_exit_code"
  {
    echo "[$agg finish] $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "[$agg exit_code] $code"
    tail -n 5 "$RESULT_DIR/${agg}_progress.log" 2>/dev/null || true
  } | tee -a "$RESULT_DIR/master.log"
  return "$code"
}

set +e
run_one refol
REFOL_CODE=$?
if [ "$REFOL_CODE" -ne 0 ]; then
  echo failed_refol > "$RESULT_DIR/status"
  echo "$REFOL_CODE" > "$RESULT_DIR/exit_code"
  exit "$REFOL_CODE"
fi
run_one fedostc
FEDOSTC_CODE=$?
if [ "$FEDOSTC_CODE" -ne 0 ]; then
  echo failed_fedostc > "$RESULT_DIR/status"
  echo "$FEDOSTC_CODE" > "$RESULT_DIR/exit_code"
  exit "$FEDOSTC_CODE"
fi
set -e
echo completed > "$RESULT_DIR/status"
echo 0 > "$RESULT_DIR/exit_code"
echo "[finish] $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$RESULT_DIR/master.log"
exit 0
