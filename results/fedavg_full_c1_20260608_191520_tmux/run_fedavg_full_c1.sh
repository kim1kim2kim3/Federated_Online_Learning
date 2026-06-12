#!/usr/bin/env bash
set +euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
echo "START $(date -Is)" | tee master.log
/home/y/Paper_2/Federated_Online_Learning/.venv/bin/python -X faulthandler run.py \
  --agg_model fedavg \
  --fedavg_client_fraction 1.0 \
  --rounds 0 \
  > fedavg_stdout.log \
  2> fedavg_stderr.log
code=$?
echo "$code" > exit_code
echo "END $(date -Is) exit_code=$code" | tee -a master.log
