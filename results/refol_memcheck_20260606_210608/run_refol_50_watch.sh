#!/usr/bin/env bash
set -euo pipefail
cd /home/y/Paper_2/Federated_Online_Learning
RUN_DIR="$1"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
rm -f PEMS-BAY_12.xlsx PEMS-BAY_12.csv
(
  while true; do
    date -Is
    free -m | sed -n '2p'
    ps -eo pid,%mem,rss,cmd | grep 'python -u run.py --agg_model refol' | grep -v grep || true
    sleep 5
  done
) > "$RUN_DIR/mem.log" 2>&1 &
WATCH=$!
set +e
/usr/bin/time -v .venv/bin/python -u run.py --agg_model refol --rounds 50 > "$RUN_DIR/refol50.raw.log" 2>&1
STATUS=$?
set -e
kill "$WATCH" 2>/dev/null || true
grep -E '^\*\*\*\* Round |^prediction rmse is:|Command terminated|Elapsed \(wall clock\)|Maximum resident set size|Exit status|Traceback|RuntimeError|Killed' "$RUN_DIR/refol50.raw.log" > "$RUN_DIR/refol50.progress.log" || true
if [ -f PEMS-BAY_12.xlsx ]; then mv PEMS-BAY_12.xlsx "$RUN_DIR/refol50.xlsx"; fi
echo "$STATUS" > "$RUN_DIR/status"
