#!/usr/bin/env bash
set -u
RUN_DIR="/home/y/Paper_2/Federated_Online_Learning/results/fedavg_full_c01_20260609_015859_tmux"
SESSION="fedavg_c01_015859"
TOTAL_ROUNDS=52093
INTERVAL_SECONDS=900
LOG="$RUN_DIR/monitor.log"
JSONL="$RUN_DIR/monitor.jsonl"
latest_round() { grep -o 'Round [0-9]*/[0-9]*' "$RUN_DIR/fedavg_stdout.log" 2>/dev/null | tail -1 | sed -E 's/Round ([0-9]+)\/([0-9]+)/\1/'; }
latest_rmse() { grep 'prediction rmse is:' "$RUN_DIR/fedavg_stdout.log" 2>/dev/null | tail -1 | sed -E 's/.*prediction rmse is: *([^ ]+).*/\1/'; }
python_pid() { ps -eo pid,args | awk '/[p]ython .*run.py --agg_model fedavg --fedavg_client_fraction 0.1 --rounds 0/ {print $1; exit}'; }
is_tmux_alive() { tmux has-session -t "$SESSION" 2>/dev/null; }
classify_status() {
  local pid="$1" round="$2" exit_code=""
  [ -f "$RUN_DIR/exit_code" ] && exit_code="$(cat "$RUN_DIR/exit_code" 2>/dev/null | tr -d '\n' || true)"
  if [ -n "$pid" ] || is_tmux_alive; then echo RUNNING
  elif [ "$exit_code" = "0" ] && [ "${round:-0}" -ge "$TOTAL_ROUNDS" ]; then echo COMPLETED
  elif [ "$exit_code" = "0" ]; then echo COMPLETED_UNVERIFIED_ROUND
  elif [ -n "$exit_code" ]; then echo "FAILED_EXIT_${exit_code}"
  elif [ "${round:-0}" -ge "$TOTAL_ROUNDS" ]; then echo ENDED_AFTER_LAST_ROUND_NO_EXIT_CODE
  else echo DIED_MID_RUN_NO_EXIT_CODE
  fi
}
write_check() {
  local now pid round rmse status rss_mb etime exit_code xlsx_count
  now="$(date -Is)"; pid="$(python_pid || true)"; round="$(latest_round || true)"; rmse="$(latest_rmse || true)"; status="$(classify_status "$pid" "${round:-0}")"
  exit_code=""; [ -f "$RUN_DIR/exit_code" ] && exit_code="$(cat "$RUN_DIR/exit_code" 2>/dev/null | tr -d '\n' || true)"
  xlsx_count="$(find "$RUN_DIR" -maxdepth 1 -type f -name '*.xlsx' 2>/dev/null | wc -l | tr -d ' ')"
  rss_mb=""; etime=""
  if [ -n "$pid" ]; then rss_mb="$(ps -p "$pid" -o rss= 2>/dev/null | awk '{printf "%.1f", $1/1024}')"; etime="$(ps -p "$pid" -o etime= 2>/dev/null | awk '{$1=$1; print}')"; fi
  echo "[$now] status=$status pid=${pid:-none} tmux=$SESSION latest_round=${round:-none}/$TOTAL_ROUNDS latest_rmse=${rmse:-none} rss_mb=${rss_mb:-none} etime=${etime:-none} exit_code=${exit_code:-none} xlsx_count=$xlsx_count" >> "$LOG"
  printf '{"time":"%s","status":"%s","pid":"%s","tmux_session":"%s","latest_round":"%s","total_rounds":%s,"latest_rmse":"%s","rss_mb":"%s","etime":"%s","exit_code":"%s","xlsx_count":%s}\n' "$now" "$status" "${pid:-}" "$SESSION" "${round:-}" "$TOTAL_ROUNDS" "${rmse:-}" "${rss_mb:-}" "${etime:-}" "${exit_code:-}" "$xlsx_count" >> "$JSONL"
  [ "$status" = RUNNING ] && return 0
  echo "[$now] FINAL: status=$status latest_round=${round:-none}/$TOTAL_ROUNDS exit_code=${exit_code:-none}" >> "$LOG"
  return 1
}
echo "Monitor started $(date -Is), interval=${INTERVAL_SECONDS}s, run_dir=$RUN_DIR, session=$SESSION" >> "$LOG"
while true; do write_check || break; sleep "$INTERVAL_SECONDS"; done
