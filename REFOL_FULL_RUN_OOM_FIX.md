# REFOL/FedOSTC Full-Round 실행 OOM 대응 기록

작성일: 2026-06-06  
대상 실행: `PEMS-BAY`, `num_clients=50`, `pred_steps=12`, full rounds `52093`

## 요약

사용자가 말한 "OOD" 현상은 실제로는 **OOM(Out Of Memory) kill**이었다.

커널 로그에서 이전 full-run Python 프로세스가 약 **27GB RSS**까지 증가한 뒤 종료된 것이 확인됐다.

```text
Out of memory: Killed process ... python
Maximum resident set size: ~27GB
```

원인은 REFOL delayed-history 저장 경로에서 `TensorDataset` slice/view를 `deepcopy()`하면서, 작은 1-row slice가 아니라 **원본 전체 tensor storage가 복제**된 것이다.

## 증상

기존 full-run 명령:

```bash
.venv/bin/python run.py --agg_model refol --rounds 0
```

문제:

- 전체 라운드 수가 `52093`로 큼
- REFOL은 delay 이후 각 client가 delayed history를 저장함
- round 13 근처에서 메모리 사용량이 급증
- Python 프로세스가 OOM kill됨
- 결과 파일이 정상적으로 생성되지 않음

## 직접 원인

기존 `client_oa.py`에는 delayed drift 비교용 history 저장이 다음처럼 되어 있었다.

```python
self.h_client_dataset = deepcopy(self.train_dataset_delayed)
```

그런데 `BaseFLServer.update_train_data()`가 만드는 `train_dataset_delayed`는 전체 데이터 tensor에서 잘라낸 slice/view다.

예:

```python
self.data["x"][train_idx:train_idx + 1, :, client_i:client_i + 1, :]
```

이 slice는 모양은 작아도 내부적으로는 원본 tensor storage를 공유할 수 있다. 따라서 `deepcopy(TensorDataset)`가 작은 visible slice만 복사하지 않고, 원본 대형 storage까지 복제할 수 있다.

결과적으로:

- client별 delayed history 저장
- round별 반복
- full PEMS-BAY tensor storage 복제
- 메모리 폭증
- OOM kill

## 수정 1 — delayed history dataset compact clone

파일: `client_oa.py`

추가한 함수:

```python
def compact_history_dataset(dataset):
    if isinstance(dataset, TensorDataset):
        return TensorDataset(
            *(tensor.detach().clone() for tensor in dataset.tensors)
        )
    return deepcopy(dataset)
```

변경:

```python
self.h_client_dataset = compact_history_dataset(self.train_dataset_delayed)
```

효과:

- 원본 full storage를 복제하지 않음
- 현재 delayed sample에 보이는 작은 tensor만 clone
- REFOL drift 비교용 history semantics는 유지
- full-run 메모리 폭증 제거

회귀 테스트:

```text
tests/test_client_history_dataset_compaction.py
```

검증 내용:

- compact history tensor가 원본 storage pointer를 공유하지 않음
- visible slice 값만 보존
- 원본 tensor를 변경해도 compact history가 바뀌지 않음

## 수정 2 — REFOL aggregation 메모리 안정화

파일: `fl-server/Refol.py`

REFOL aggregation에서 다음을 적용했다.

- caller의 `local_states`를 직접 mutate하지 않음
- Python `tolist()` 기반 대형 임시 list 생성을 피하고 `torch.cat` / `torch.stack` 사용
- aggregation GCN은 optimizer/backward 대상이 아니므로 `torch.no_grad()` 사용
- aggregated global model은 `detach().cpu().clone()`으로 저장

핵심 목적:

- 서버 aggregation 결과가 autograd graph를 붙잡지 않게 함
- full-round 반복 중 graph/history가 누적되지 않게 함
- CPU tensor state로 정리해 다음 round에 넘김

회귀 테스트:

```text
tests/test_refol_aggregation_memory.py
```

검증 내용:

- aggregation 후 `server.global_model` tensor가 `requires_grad=False`
- `grad_fn is None`
- caller의 `local_states` 길이가 변하지 않음
- global state가 CPU tensor로 저장됨

## 수정 3 — full-run 전용 streaming runner 추가

파일:

```text
scripts/run_full_baseline_stream.py
```

기존 `run.py`는 다음 이유로 full-run에 불리했다.

- round별 stdout이 많음
- xlsx writer를 끝까지 들고 감
- 프로세스가 죽으면 결과 파일이 온전히 남지 않을 수 있음

새 runner는 다음 방식이다.

```bash
.venv/bin/python scripts/run_full_baseline_stream.py \
  --agg_model refol \
  --rounds 0 \
  --output_dir <result_dir> \
  --progress_every 500
```

동작:

- round별 metric을 CSV에 즉시 write
- progress log에 RSS/elapsed/sec_per_round 기록
- verbose client 로그는 `/dev/null`로 redirect
- `gc.collect()`를 progress interval마다 호출
- 프로세스가 중간에 죽어도 partial CSV가 남음

생성 파일:

```text
refol_metrics.csv
refol_progress.log
refol_summary.json
fedostc_metrics.csv
fedostc_progress.log
fedostc_summary.json
combined_summary.json
```

## 검증 결과

### 단위/회귀 테스트

다음 검증을 통과했다.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

결과:

```text
Ran 63 tests
OK
```

### REFOL 20-round smoke

delay 이후 path까지 포함한 REFOL 20-round smoke를 실행했다.

결과 요약:

```text
rounds_completed: 20
mean_rmse: 65.6320
mean_mae: 65.6130
runtime rss_mb: ~1.63GB
time -v maximum resident set size: ~4.8GB
exit status: 0
```

이전처럼 round 13 부근에서 27GB까지 치솟는 현상은 재현되지 않았다.

## 현재 full-run 실행 방식

REFOL과 FedOSTC는 동시에 실행하지 않고 **순차 실행**한다.

이유:

- 두 baseline을 동시에 돌리면 메모리 압박이 커짐
- REFOL 완료 후 FedOSTC를 자동 실행하는 것이 안전함

현재 실행은 tmux 세션에서 분리 실행 중이다.

예:

```text
tmux session: fedfull_211927
result_dir: /home/y/Paper_2/Federated_Online_Learning/results/full_rounds_tmux_20260606_211927
```

진행 확인:

```bash
RESULT_DIR=$(cat /tmp/fedfull_latest_dir)
tail -5 "$RESULT_DIR/refol_progress.log"
wc -l "$RESULT_DIR/refol_metrics.csv"
free -h
```

## 수정 후 기대 상태

정상 상태:

- REFOL RSS가 약 1.6GB 수준에서 안정적으로 유지
- `refol_metrics.csv` row가 계속 증가
- REFOL 완료 후 FedOSTC가 자동 시작
- 최종적으로 `combined_summary.json` 생성

비정상 상태:

- RSS가 다시 수십 GB까지 증가
- progress log가 멈춤
- tmux session이 사라짐
- `*_stderr.log`에 traceback 또는 OOM kill 흔적 발생

## 변경 파일 목록

OOM 대응 직접 관련:

```text
client_oa.py
fl-server/Refol.py
scripts/run_full_baseline_stream.py
tests/test_client_history_dataset_compaction.py
tests/test_refol_aggregation_memory.py
tests/test_client_local_execute_load_order.py
```

비고:

- `tests/test_client_local_execute_load_order.py`는 새 `torch.utils.data.TensorDataset` import를 stub 환경에서도 통과시키기 위해 mock module을 보강했다.
- FedOSTC 평가 방식 정합성 변경과 별개로, 이 문서는 full-run OOM 대응에 초점을 둔다.
