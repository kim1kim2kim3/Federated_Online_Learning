# REFOL vs FedOSTC Full-Run 결과 종합

작성일: 2026-06-08  
대상 실행: `PEMS-BAY`, `num_clients=50`, `pred_steps=12`, full rounds `52093`

## 요약

같은 full-run 조건에서 **FedOSTC가 예측 성능은 더 좋고, REFOL은 실행 속도가 더 빠르다.**

두 알고리즘 모두 OOM 없이 full-run을 완료했다.

- REFOL: CPU/RAM OOM 대응 후 full-run 완료
- FedOSTC: GPU OOM 대응 후 full-run 완료
- FedOSTC GPU memory는 warm-up 이후 마지막 round까지 plateau를 유지했다.

## 실행 조건

```text
dataset: PEMS-BAY
num_clients: 50
pred_steps: 12
rounds: 52093
```

결과 파일:

```text
REFOL summary:   results/full_rounds_tmux_20260606_211927/refol_summary.json
FedOSTC summary: /home/y/Paper_2/Federated_Online_Learning/results/fedostc_full_20260607_214117/fedostc_summary.json
REFOL metrics:   /home/y/Paper_2/Federated_Online_Learning/results/full_rounds_tmux_20260606_211927/refol_metrics.csv
FedOSTC metrics: /home/y/Paper_2/Federated_Online_Learning/results/fedostc_full_20260607_214117/fedostc_metrics.csv
```

## 주요 결과 비교

| 항목 | REFOL | FedOSTC | 해석 |
|---|---:|---:|---|
| 완료 rounds | 52,093 | 52,093 | 둘 다 full-run 완료 |
| mean RMSE | 6.745039 | **4.768749** | FedOSTC 약 **29.30% 낮음** |
| mean MAE | 5.932780 | **4.460302** | FedOSTC 약 **24.82% 낮음** |
| last RMSE | 3.983003 | **2.032607** | FedOSTC 약 **48.97% 낮음** |
| last MAE | 2.598688 | **1.878323** | FedOSTC 약 **27.72% 낮음** |
| best RMSE | 1.575994 @ round 1886 | **0.937913 @ round 280** | FedOSTC 약 **40.49% 낮음** |
| best MAE at best-RMSE round | 1.396564 | **0.871913** | FedOSTC 약 **37.57% 낮음** |
| 실행 시간 | **5h 12m 22s** | 8h 40m 20s | REFOL이 더 빠름 |
| sec/round | **0.359785s** | 0.599305s | REFOL이 더 빠름 |
| RSS | 1655.438 MB | 1715.941 MB | 둘 다 안정적 |

## REFOL 결과

```json
{
  "rounds_completed": 52093,
  "mean_rmse": 6.745038837421566,
  "mean_mae": 5.932780162375002,
  "last": {
    "round": 52093,
    "rmse": 3.9830029010772705,
    "mae": 2.5986881256103516
  },
  "best_rmse": {
    "round": 1886,
    "rmse": 1.5759942531585693,
    "mae": 1.3965643644332886
  },
  "agg_model": "refol",
  "metrics_path": "/home/y/Paper_2/Federated_Online_Learning/results/full_rounds_tmux_20260606_211927/refol_metrics.csv",
  "progress_path": "/home/y/Paper_2/Federated_Online_Learning/results/full_rounds_tmux_20260606_211927/refol_progress.log",
  "elapsed_sec": 18742.277,
  "rss_mb": 1655.438
}
```

요약:

- 완료 rounds: `52093`
- mean RMSE: `6.745038837421566`
- mean MAE: `5.932780162375002`
- last RMSE/MAE: `3.9830029010772705` / `2.5986881256103516`
- best RMSE: `1.5759942531585693` at round `1886`
- 실행 시간: `5h 12m 22s`
- RSS: `1655.438 MB`

## FedOSTC 결과

```json
{
  "rounds_completed": 52093,
  "mean_rmse": 4.768749369333132,
  "mean_mae": 4.460302384386704,
  "last": {
    "round": 52093,
    "rmse": 2.032607316970825,
    "mae": 1.8783233165740967
  },
  "best_rmse": {
    "round": 280,
    "rmse": 0.9379131197929382,
    "mae": 0.8719134330749512
  },
  "agg_model": "fedostc",
  "metrics_path": "/home/y/Paper_2/Federated_Online_Learning/results/fedostc_full_20260607_214117/fedostc_metrics.csv",
  "progress_path": "/home/y/Paper_2/Federated_Online_Learning/results/fedostc_full_20260607_214117/fedostc_progress.log",
  "elapsed_sec": 31219.604,
  "rss_mb": 1715.941,
  "cuda_allocated_mb": 37.661,
  "cuda_reserved_mb": 64.0,
  "cuda_max_allocated_mb": 46.645
}
```

요약:

- 완료 rounds: `52093`
- mean RMSE: `4.768749369333132`
- mean MAE: `4.460302384386704`
- last RMSE/MAE: `2.032607316970825` / `1.8783233165740967`
- best RMSE: `0.9379131197929382` at round `280`
- 실행 시간: `8h 40m 20s`
- RSS: `1715.941 MB`
- CUDA allocated: `37.661 MB`
- CUDA reserved: `64.0 MB`
- CUDA max allocated: `46.645 MB`

## FedOSTC GPU 안정성 확인

FedOSTC full-run 마지막 구간 progress log에서 GPU memory는 다음과 같이 유지됐다.

```text
round 51200: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 51300: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 51400: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 51500: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 51600: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 51700: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 51800: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 51900: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 52000: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 52093: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
```

확인 결과:

- full-run 전체 `52093` rounds 완료
- `fedostc_metrics.csv` rows: `52094` = header + `52093` rounds
- exit code: `0`
- CUDA memory가 round 수에 비례해 증가하지 않음
- GPU OOM 없이 정상 완료

## 해석

### 정확도 관점

FedOSTC가 REFOL보다 더 좋은 결과를 냈다.

- mean RMSE: FedOSTC가 REFOL 대비 약 **29.30% 감소**
- mean MAE: FedOSTC가 REFOL 대비 약 **24.82% 감소**
- final/last metric도 FedOSTC가 더 낮음
- best RMSE도 FedOSTC가 더 낮음

### 속도 관점

REFOL이 더 빠르다.

- REFOL: `0.359785s/round`
- FedOSTC: `0.599305s/round`
- FedOSTC는 REFOL보다 약 **66.57% 더 오래 걸림**

### 안정성 관점

둘 다 full-run 안정화에 성공했다.

- REFOL은 CPU/RAM OOM 문제를 해결한 뒤 full-run 완료
- FedOSTC는 GPU OOM 문제를 해결한 뒤 full-run 완료
- FedOSTC GPU memory는 마지막까지 plateau 상태 유지

## 결론

- **성능 우선**이면 FedOSTC가 더 유리하다.
- **속도/효율 우선**이면 REFOL이 더 유리하다.
- OOM 안정성 측면에서는 두 baseline 모두 full-run 완료로 검증됐다.

현재 실험 기준 최종 결론은 다음과 같다.

```text
FedOSTC: 더 낮은 RMSE/MAE, 더 긴 실행 시간
REFOL: 더 빠른 실행, 더 높은 RMSE/MAE
```
