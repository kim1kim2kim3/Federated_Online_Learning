#!/usr/bin/env bash
set -euo pipefail
cd /home/y/Paper_2/Federated_Online_Learning
RUN_DIR="$1"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run_one() {
  local model="$1"
  local out_xlsx="$RUN_DIR/${model}_PEMS-BAY_12.xlsx"
  local raw_log="$RUN_DIR/${model}.raw.log"
  local log="$RUN_DIR/${model}.progress.log"
  local status_file="$RUN_DIR/${model}.status"
  rm -f PEMS-BAY_12.xlsx PEMS-BAY_12.csv "$out_xlsx" "$RUN_DIR/${model}_PEMS-BAY_12.csv"
  echo "START $(date -Is) model=$model" | tee "$status_file"
  set +e
  /usr/bin/time -v .venv/bin/python -u run.py --agg_model "$model" --rounds 0 > "$raw_log" 2>&1
  local status=$?
  set -e
  grep -E '^\*\*\*\* Round |^prediction rmse is:|Traceback|RuntimeError|OutOfMemory|out of memory|Killed|Command terminated|Elapsed \(wall clock\)|Maximum resident set size|Exit status' "$raw_log" > "$log" || true
  if [ -f PEMS-BAY_12.xlsx ]; then mv PEMS-BAY_12.xlsx "$out_xlsx"; fi
  if [ -f PEMS-BAY_12.csv ]; then mv PEMS-BAY_12.csv "$RUN_DIR/${model}_PEMS-BAY_12.csv"; fi
  echo "END $(date -Is) model=$model status=$status" | tee -a "$status_file"
  return "$status"
}
run_one refol
run_one fedostc
python - <<'PY' "$RUN_DIR"
import json, math, statistics, sys
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
run_dir = Path(sys.argv[1])
ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def shared(zf):
    try:
        root = ET.fromstring(zf.read('xl/sharedStrings.xml'))
    except KeyError:
        return []
    vals = []
    for si in root.findall('x:si', ns):
        vals.append(''.join(t.text or '' for t in si.findall('.//x:t', ns)))
    return vals
def parse_xlsx(path):
    with ZipFile(path) as zf:
        ss = shared(zf)
        root = ET.fromstring(zf.read('xl/worksheets/sheet1.xml'))
    rows = []
    for row in root.findall('.//x:sheetData/x:row', ns):
        vals = []
        for c in row.findall('x:c', ns):
            v = c.find('x:v', ns)
            if v is None:
                vals.append(None)
            elif c.attrib.get('t') == 's':
                vals.append(ss[int(v.text)])
            else:
                vals.append(float(v.text))
        rows.append(vals)
    header, data = rows[0], rows[1:]
    return header, data
def summarize(model):
    path = run_dir / f'{model}_PEMS-BAY_12.xlsx'
    header, rows = parse_xlsx(path)
    rmse = [float(r[1]) for r in rows if len(r) > 1 and r[1] is not None and math.isfinite(float(r[1]))]
    mae = [float(r[2]) for r in rows if len(r) > 2 and r[2] is not None and math.isfinite(float(r[2]))]
    return {
        'rows': len(rows),
        'rmse_mean': statistics.fmean(rmse),
        'rmse_last': rmse[-1],
        'rmse_min': min(rmse),
        'rmse_max': max(rmse),
        'mae_mean': statistics.fmean(mae),
        'mae_last': mae[-1],
        'mae_min': min(mae),
        'mae_max': max(mae),
    }
summary = {m: summarize(m) for m in ('refol', 'fedostc')}
(run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
PY
