# FedOSTC Full-Round GPU OOM 대응 기록

작성일: 2026-06-07  
대상 실행: `PEMS-BAY`, `num_clients=50`, `pred_steps=12`, full rounds `52093`

## 요약

FedOSTC full-run 실패 원인은 CPU RSS 폭증이 아니라 **GPU OOM**이었다.

핵심 원인은 FedOSTC 서버가 매 round마다 다음 장기 history에 model/prediction state를 계속 보존하면서 GPU tensor snapshot이 누적된 것이다.

```text
w_pred[r]
w_update[tau]
prediction_history[r]
rho_history[tau]
```

이번 조치는 FedOSTC metric 산식, delayed-label update 순서, aggregation 식을 바꾸는 성능 개선 패치가 아니다. 목적은 **알고리즘에 필요하지 않은 provenance/history retention을 bounded 또는 CPU-only로 바꿔 GPU 메모리 선형 증가를 제거**하는 것이다.

## REFOL OOM 조치와의 관계

REFOL 조치와 같은 계열의 메모리 안정화 작업이지만 직접 원인은 다르다.

- REFOL: `TensorDataset` slice/view를 `deepcopy()`하면서 원본 대형 tensor storage가 복제된 CPU/RAM OOM이 주된 원인.
- FedOSTC: round별 CUDA model state와 prediction history가 장기 보존된 GPU OOM이 주된 원인.

공통점은 둘 다 학습 알고리즘을 바꾸지 않고 `detach`, `cpu`, bounded history, graph/state retention 제거로 full-run 안정성을 높였다는 점이다.

## 증상

기존 FedOSTC full-run/long-run에서 다음 문제가 발생했다.

- round가 증가할수록 GPU memory allocated/reserved가 계속 증가
- 진행 중 CUDA OOM 발생
- partial CSV는 남을 수 있지만 model state checkpoint가 없어서 정확한 재개가 어려움
- 따라서 안정화 후 FedOSTC는 partial 결과를 이어붙이지 않고 새 결과 디렉터리에서 처음부터 재실행하는 것이 기본 방침

## 직접 원인

### 1. `w_pred[r]` 누적

`predict_current()`에서 prediction round의 global state snapshot을 `w_pred[r]`에 저장했다.

기존 의도상 `w_pred`는 prediction provenance/debug 기록이다. Delayed training 초기값으로 쓰이지 않는다.

FedOSTC contract도 다음을 요구한다.

- delayed training은 `w_pred[tau]`에서 시작하지 않음
- update는 현재 update model state에서 수행
- `w_pred[r]`는 provenance only

따라서 full snapshot을 모든 round에 GPU로 보존할 필요가 없다.

### 2. `w_update[tau]` 누적

`update_from_feedback(tau)`에서 delayed local training의 초기 update model을 `w_update[tau]`에 저장했다.

이 값은 해당 `tau` update를 시작할 때 사용된 기록이며, update가 끝난 뒤 다음 recurrence에 다시 사용되지 않는다. 모든 `tau`의 full model state를 GPU에 계속 보존하면 메모리가 round 수에 비례해 증가한다.

### 3. `prediction_history[r]` pending window 초과 보존

`prediction_history[r]`는 다음 두 용도에 필요하다.

1. 현재 round `r`의 metric 계산: `ŷ_r` vs `y_r`
2. label delay 이후 feedback update: `tau = r - delay`

`tau` update가 끝난 prediction은 더 이상 필요하지 않다. 따라서 pending window만 유지하면 된다.

### 4. `rho_history[tau]` lookup window 초과 보존

`rho_history`는 period aggregation에서 과거 `rho_key = tau + 1 - period_steps`를 찾기 위해 필요하다.

현재 기본 `period_steps=288`에서는 전체 full-run history가 아니라 period lookup에 필요한 window만 보존하면 된다.

## 수정 1 — FedOSTC state snapshot을 CPU-only로 변경

파일: `fl-server/Fedostc.py`

`clone_state()` 기본 동작을 CPU detached copy로 변경했다.

```python
def clone_state(
    state: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str | None = "cpu",
) -> dict[str, torch.Tensor]:
    target = torch.device(device) if device is not None else None
    return {
        name: (
            tensor.detach().to(target, copy=True)
            if target is not None
            else tensor.detach().clone()
        )
        for name, tensor in state.items()
    }
```

효과:

- `global_model` 장기 state가 CPU tensor로 저장됨
- `w_pred`, `w_update`, local aggregation state도 기본적으로 CPU tensor가 됨
- client/model 실행 시 `load_state_dict()`와 `.to(self.device)` 경로에서 필요한 순간에만 GPU로 올라감
- CUDA graph/state를 장기 dict가 붙잡지 않음

## 수정 2 — `w_pred` / `w_update` bounded retention

파일: `fl-server/Fedostc.py`

추가된 helper:

```python
def _state_history_keep(self) -> int:
    return max(0, int(getattr(self.config, "fedostc_state_history_keep", 1)))

def _record_state_snapshot(self, history, key, state) -> None:
    keep = self._state_history_keep()
    if keep <= 0:
        return
    history[int(key)] = clone_state(state)
    self._evict_oldest(history, keep)
```

기본값:

```text
fedostc_state_history_keep = 1
```

효과:

- provenance/debug용 최근 snapshot만 보존
- full-run에서 model-size × round-count 형태의 GPU/CPU snapshot 누적 방지
- 필요하면 config로 `0`을 지정해 provenance snapshot 자체를 끌 수 있음

## 수정 3 — `prediction_history` pending window eviction

파일: `fl-server/Fedostc.py`

현재 prediction 저장 후 old history를 bounded로 유지한다.

```python
self.prediction_history[int(rround)] = predictions
self._evict_oldest(self.prediction_history, self._prediction_history_keep())
```

feedback update가 끝난 `tau`는 즉시 제거한다.

```python
def _evict_feedback_history(self, tau: int) -> None:
    self.prediction_history.pop(int(tau), None)
    self._evict_oldest(self.prediction_history, self._prediction_history_keep())
```

보존 크기:

```python
def _prediction_history_keep(self) -> int:
    delay = getattr(self, "delay", None)
    if delay is None:
        delay = self._resolve_label_delay()
    return max(1, int(delay) + 1)
```

효과:

- 현재 평가와 delayed feedback에 필요한 pending predictions만 보존
- `tau` update 완료 후 prediction tensor 참조 제거
- flush 시 남은 pending predictions만 처리하고 제거

## 수정 4 — `rho_history` period lookup window eviction

파일: `fl-server/Fedostc.py`

`aggregate_for_tau()`에서 현재 `rho`를 계산한 뒤 period lookup에 더 이상 필요 없는 과거 key를 제거한다.

```python
def _evict_rho_history(self, tau: int) -> None:
    if self.period_steps <= 0:
        return
    next_required_key = int(tau) + 2 - int(self.period_steps)
    for key in list(self.rho_history):
        if key < next_required_key:
            del self.rho_history[key]
```

효과:

- `period_steps=288` 기준 필요한 lookup window만 유지
- Eq.15 period aggregation에서 필요한 `rho_key`는 보존
- full-run 전체 tau의 rho를 계속 쌓지 않음

## 수정 5 — round-local CUDA tensor 참조 정리

파일: `fl-server/Fedostc.py`

`predict_current()`와 `update_from_feedback()`에서 round-local tensor 참조를 정리했다.

```python
del hidden, refined
del delayed_hidden, delayed_refined, local_states
```

효과:

- 현재 round의 hidden/refined tensor가 장기 history로 남지 않음
- Python reference 때문에 CUDA tensor lifetime이 불필요하게 길어지는 것을 방지

## 수정 6 — FedOSTC client state snapshot CPU-only

파일: `client_fedostc.py`

client의 장기 state snapshot도 CPU detached clone으로 저장한다.

```python
def _snapshot_model_state(self) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in self.model.state_dict().items()
    }
```

적용 위치:

- `__init__()`의 `self.state_dict`
- `load_model_state()` 이후 `self.state_dict`
- `train_delayed()` 결과 `self.last_result["state_dict"]`

효과:

- client의 `state_dict`, `last_result`, `local_result`가 CUDA tensor를 장기 보존하지 않음
- server가 local 결과를 수집할 때도 CPU state 중심으로 aggregation 수행

## 수정 7 — streaming runner GPU memory logging / guard

파일: `scripts/run_full_baseline_stream.py`

기본 allocator 환경 설정:

```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

progress/summary에 CUDA memory 필드를 추가했다.

```text
cuda_allocated_mb
cuda_reserved_mb
cuda_max_allocated_mb
```

추가 옵션:

```bash
--cuda_memory_guard_mb <MB>
```

동작:

- guard 값이 `0`이면 비활성화
- CUDA reserved memory가 threshold를 넘으면 현재 round metric을 기록한 뒤 안전하게 abort event를 남기고 종료
- abort 시 exit code는 `2`

예:

```bash
.venv/bin/python scripts/run_full_baseline_stream.py \
  --agg_model fedostc \
  --rounds 1000 \
  --progress_every 100 \
  --cuda_memory_guard_mb 2048 \
  --output_dir /tmp/fedostc_stream_smoke_1000
```

## 성능/알고리즘 영향 평가

이번 조치는 예측 성능을 의도적으로 바꾸지 않는다.

유지되는 계약:

- metric은 현재 prediction `ŷ_r`와 현재 label `y_r`로 계산
- delayed update는 `tau = r - delay`
- flush는 남은 delayed update만 수행하고 metric row를 추가하지 않음
- delayed training은 `w_pred[tau]`로 초기화하지 않음
- aggregation 식과 `rho` 계산 방식은 유지

짧은 A/B 확인:

- current 패치 vs old-like GPU snapshot 방식
- 동일 seed, 30 rounds
- max RMSE diff: `1.9073486328125e-06`
- max MAE diff: `1.9073486328125e-06`

이는 CPU/GPU copy 위치 차이에서 발생할 수 있는 float 오차 수준이다. 따라서 객관적으로 이번 조치는 **모델 성능 패치가 아니라 메모리 retention 패치**다.

단, 실행 성능 관점에서는 장기 state를 CPU에 보관하기 때문에 일부 state load/copy 비용이 생길 수 있다. 그 대신 GPU memory가 round 수에 비례해 증가하지 않는다.

## 회귀 테스트

### FedOSTC history bounded retention

파일: `tests/test_fedostc_phase4_server.py`

검증 내용:

- 여러 round 이후 `prediction_history` 크기가 delay window 이상 증가하지 않음
- `w_pred` 크기가 1 이하
- `w_update` 크기가 1 이하
- `rho_history` 크기가 `period_steps` 이하
- flush 이후 pending `prediction_history`가 비워짐

### long-lived CUDA tensor retention 차단

파일: `tests/test_fedostc_phase4_server.py`

CUDA 가능 환경에서 검증 내용:

- `global_model`
- `w_pred`
- `w_update`
- `prediction_history`
- `rho_history`

위 장기 보존 객체에 CUDA tensor가 남지 않음.

### client result CPU state

파일: `tests/test_fedostc_phase3_client.py`

검증 내용:

- `train_delayed()` 결과 `state_dict`가 CUDA tensor를 보존하지 않음
- client `state_dict`도 CUDA tensor를 보존하지 않음

## 검증 결과

### 단위/회귀 테스트

최종 전체 테스트:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

결과:

```text
Ran 65 tests
OK
```

### FedOSTC 1000-round GPU smoke

명령:

```bash
.venv/bin/python scripts/run_full_baseline_stream.py \
  --agg_model fedostc \
  --rounds 1000 \
  --progress_every 100 \
  --cuda_memory_guard_mb 2048 \
  --output_dir /tmp/fedostc_stream_smoke_1000
```

결과 요약:

```text
rounds_completed: 1000
mean_rmse: 4.410963970482349
mean_mae: 4.1266165378093715
last round: 1000
last rmse: 2.527803421020508
last mae: 2.3885908126831055
elapsed_sec: 588.664
exit status: 0
```

GPU memory progress:

```text
round 100:  cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 200:  cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 300:  cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 400:  cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 500:  cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 600:  cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 700:  cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 800:  cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 900:  cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
round 1000: cuda_allocated_mb=37.661, cuda_reserved_mb=64.0, cuda_max_allocated_mb=46.645
```

확인 결과:

- 1000 rounds 완료
- metric CSV 정상 생성
- GPU allocated/reserved/max가 round 수에 비례해 증가하지 않고 round 100 이후 plateau 유지

### guard 동작 smoke

명령:

```bash
.venv/bin/python scripts/run_full_baseline_stream.py \
  --agg_model fedostc \
  --rounds 1 \
  --progress_every 1 \
  --cuda_memory_guard_mb 1 \
  --output_dir /tmp/fedostc_stream_guard
```

결과:

```text
exit=2
abort_reason: cuda_memory_guard_exceeded
round: 1
```

즉 guard가 threshold 초과 시 progress/summary에 abort 정보를 남기고 안전하게 종료한다.

## 현재 full-run 권장 실행 방식

FedOSTC는 안정화 후 partial 결과를 이어붙이지 않고 새 결과 디렉터리에서 full-run을 처음부터 실행한다.

예:

```bash
RESULT_DIR="results/fedostc_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

.venv/bin/python scripts/run_full_baseline_stream.py \
  --agg_model fedostc \
  --rounds 0 \
  --progress_every 500 \
  --cuda_memory_guard_mb 2048 \
  --output_dir "$RESULT_DIR"
```

진행 확인:

```bash
tail -5 "$RESULT_DIR/fedostc_progress.log"
wc -l "$RESULT_DIR/fedostc_metrics.csv"
cat "$RESULT_DIR/fedostc_summary.json"
```

REFOL 결과가 같은 디렉터리에 이미 있다면 FedOSTC summary 작성 후 `combined_summary.json`도 생성된다.

## 수정 후 기대 상태

정상 상태:

- FedOSTC GPU allocated/reserved가 초반 warm-up 이후 plateau 유지
- `fedostc_metrics.csv` row가 계속 증가
- `fedostc_progress.log`에 RSS/CUDA memory가 주기적으로 기록
- 완료 후 `fedostc_summary.json` 생성
- REFOL summary가 같은 디렉터리에 있으면 `combined_summary.json` 생성

비정상 상태:

- `cuda_reserved_mb`가 round 수에 비례해 계속 증가
- progress log가 멈춤
- stderr에 CUDA OOM traceback 발생
- guard 사용 시 `abort_reason: cuda_memory_guard_exceeded` 기록

## 변경 파일 목록

FedOSTC GPU OOM 대응 직접 관련:

```text
fl-server/Fedostc.py
client_fedostc.py
scripts/run_full_baseline_stream.py
tests/test_fedostc_phase3_client.py
tests/test_fedostc_phase4_server.py
FEDOSTC_FULL_RUN_GPU_OOM_FIX.md
```

관련/비교 문서:

```text
REFOL_FULL_RUN_OOM_FIX.md
```
