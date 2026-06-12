#!/usr/bin/env bash
set -u
cd /home/y/Paper_2/Federated_Online_Learning
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONWARNINGS=ignore
RESULT_DIR="$1"
echo "START $(date -Is) result_dir=$RESULT_DIR" > "$RESULT_DIR/master.log"
for model in refol fedostc; do
  echo "MODEL_START $model $(date -Is)" >> "$RESULT_DIR/master.log"
  /usr/bin/time -v .venv/bin/python scripts/run_full_baseline_stream.py \
    --agg_model "$model" \
    --rounds 0 \
    --output_dir "$RESULT_DIR" \
    --progress_every 500 \
    > "$RESULT_DIR/${model}_stdout.log" 2> "$RESULT_DIR/${model}_stderr.log"
  status=$?
  echo "MODEL_DONE $model status=$status $(date -Is)" >> "$RESULT_DIR/master.log"
  if [ "$status" -ne 0 ]; then
    echo "STOP_AFTER_FAILURE $model" >> "$RESULT_DIR/master.log"
    exit "$status"
  fi
done
.venv/bin/python - <<'PY' "$RESULT_DIR" >> "$RESULT_DIR/master.log" 2>&1
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
combined = {}
for model in ('refol', 'fedostc'):
    path = out / f'{model}_summary.json'
    if path.exists():
        combined[model] = json.loads(path.read_text())
(out / 'combined_summary.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
print(json.dumps(combined, indent=2))
PY
echo "ALL_DONE $(date -Is)" >> "$RESULT_DIR/master.log"
