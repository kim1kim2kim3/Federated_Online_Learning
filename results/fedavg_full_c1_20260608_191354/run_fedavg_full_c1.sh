#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
/home/y/Paper_2/Federated_Online_Learning/.venv/bin/python run.py \
  --agg_model fedavg \
  --fedavg_client_fraction 1.0 \
  --rounds 0 \
  > fedavg_stdout.log \
  2> fedavg_stderr.log
echo $? > exit_code
