# Delayed-Label Federated Online Learning 시뮬레이션 프레임워크

본 프로젝트는 스트리밍 데이터에서 **지금 들어온 입력은 즉시 예측하고(predict-now), 라벨은 지연(delay) 이후 도착했을 때만 학습에 사용하는(learn-later)** 지연 라벨(Delayed Label / Delayed Feedback) 환경을 타겟으로 한 **Federated Online Learning(FOL)** 시뮬레이션 프레임워크입니다.

핵심 목표는 특정 REFOL 재현 코드에 머무르는 것이 아니라, REFOL/FedAvg/FedOSTC 같은 aggregation 알고리즘을 동일한 delayed-label online protocol 위에서 공정하게 비교할 수 있도록 만드는 것입니다. 현재 코드의 REFOL 구현은 기존 baseline 방법론을 delayed-label FOL 환경에 맞게 이식한 예시이며, FedOSTC 경로는 FedOSTC 논문 구현을 delayed-label feedback 환경에 맞게 이식한 실험 경로입니다. 서버 구조를 확장해 다른 aggregation 전략도 같은 평가 흐름에서 실험할 수 있습니다.

---

## ✨ 주요 특징 (Key Features)

1. **Delayed feedback scheduling**
   - `delay`만큼 라벨 도착을 지연시켜 현실적인 delayed feedback 스트림을 시뮬레이션합니다.
   - `default_config.yaml`의 `delay: "pred_steps"` 설정은 예측 horizon(`pred_steps`)과 같은 길이의 라벨 지연을 사용합니다.

2. **Prequential evaluation (Predict-then-Update)**
   - 각 시점의 입력은 학습보다 먼저 평가되어 온라인 환경의 `predict-now, learn-later` 순서를 보존합니다.
   - 라운드 `r` row에는 round `r`의 예측 `ŷ_r`를 current label `y_r`와 비교한 평가 지표(RMSE, MAE)를 기록합니다.
   - 학습/update에는 라벨이 도착한 과거 샘플 `tau = r - delay`만 투입하므로, 평가는 REFOL/BaseFLServer와 같은 current-round 기준을 유지하면서도 학습은 delayed feedback 순서를 따릅니다.

3. **Label leakage 방지**
   - 현재 또는 미래 라벨을 예측 입력으로 사용하지 않도록 delayed-label scheduling과 GRU 입력 경로를 분리합니다.
   - 테스트는 예측 결과가 `y`/`y_attr` 변조에 영향을 받지 않는지 확인합니다.

4. **REFOL-style drift-based selective training**
   - 입력 피처 분포 변화(KL divergence)를 이용해 concept drift를 감지하고, 변화가 큰 클라이언트를 선택적으로 학습시킵니다.
   - 지연 라벨 상황에서도 계산/통신 비용을 줄이는 resource-efficient FOL 실험을 지원합니다.

5. **확장 가능한 FedAvg/REFOL/FedOSTC 서버 구조**
   - 공통 delayed-label 실행 흐름은 `BaseFLServer`가 담당합니다.
   - `Refol.py`, `Fedavg.py`, `Fedostc.py`, `template.py`를 기반으로 aggregation 로직만 교체해 새로운 서버를 추가할 수 있습니다.
   - FedOSTC는 논문 Algorithm/Eq. 경로를 delayed scheduling과 future `y` leakage 방지 조건에 맞게 조정한 별도 서버/클라이언트/모델 경로를 사용합니다.

---

## 📂 폴더 구조 (Folder Structure)

```directory
Federated_Online_Learning/
├── config.py                   # 실행 인자 및 yaml 설정 로더
├── default_config.yaml         # pred_steps, delay, agg_model 등 기본 설정
├── run.py                      # 학습 프로세스 진입점
├── client_oa.py                # 로컬 클라이언트 객체 및 선택적 학습 로직
├── client_fedostc.py           # FedOSTC 전용 client-side encode/decode/update 로직
├── fl-server/
│   ├── base_server.py          # delayed-label FOL 공통 베이스 서버
│   ├── Refol.py                # REFOL-style aggregation 구현체
│   ├── Fedavg.py               # Federated Averaging(FedAvg) 구현체
│   ├── Fedostc.py              # FedOSTC delayed-label adaptation 구현체
│   └── template.py             # 신규 aggregation 서버 확장 템플릿
├── models/
│   ├── fl_model.py             # 클라이언트 GRU 예측 모델
│   ├── fedostc_model.py        # FedOSTC encoder/decoder 모델
│   ├── fedostc_gat.py          # FedOSTC Eq. (8)-(10) attention GAT
│   └── AggregationGCN.py       # REFOL 가중치 병합용 Attention GCN 모델
├── utils/
│   └── process_increment.py    # 데이터 로딩 및 raw-scale metric 유틸리티
├── data/
│   ├── split_data_increment.py # 원본 h5 데이터를 증분 npz 데이터로 변환
│   └── sensor_graph/           # 센서 그래프 메타데이터 및 인접 행렬
└── tests/                      # delayed-label protocol 회귀 테스트
```

---

## 💾 데이터 준비 (Data Preparation)

[`DCRNN`](https://github.com/liyaguang/DCRNN/blob/master/README.md) 저장소의 안내에 따라 교통 데이터 원본(`metr-la.h5`, `pems-bay.h5`)을 다운로드하고 `data/` 디렉토리에 배치합니다.

현재 기본 설정은 `pred_steps: 12`이므로, 같은 horizon의 증분 데이터셋(`12_series.npz`)을 생성하는 예시는 다음과 같습니다.

```bash
# 프로젝트 루트에서 실행
mkdir -p data/METR-LA data/PEMS-BAY

cd data

# METR-LA 데이터셋 처리
../.venv/bin/python split_data_increment.py \
  --pred_steps=12 \
  --output_dir=./METR-LA \
  --traffic_df_filename=./metr-la.h5

# PEMS-BAY 데이터셋 처리
../.venv/bin/python split_data_increment.py \
  --pred_steps=12 \
  --output_dir=./PEMS-BAY \
  --traffic_df_filename=./pems-bay.h5
```

다른 예측 horizon을 실험하려면 `--pred_steps`와 `default_config.yaml`의 `pred_steps`를 동일하게 맞춥니다.

---

## 🚀 실행 방법 (How to Run)

`default_config.yaml`의 `agg_model` 기본값은 빈 문자열이므로, 실행 시 비교할 aggregation 알고리즘을 명시해야 합니다.

```bash
# REFOL-style delayed-label FOL 실행
.venv/bin/python run.py --agg_model refol

# FedAvg delayed-label FOL 실행 (C=1.0, 모든 history 보유 client 참여)
.venv/bin/python run.py --agg_model fedavg

# FedAvg partial participation 실행 (C=0.1, round당 history 보유 client의 10%)
.venv/bin/python run.py --agg_model fedavg --fedavg_client_fraction 0.1

# FedOSTC delayed-label adaptation 실행
.venv/bin/python run.py --agg_model fedostc

# FedOSTC delayed update 동작 확인용 짧은 실행
.venv/bin/python run.py --agg_model fedostc --delay 1 --rounds 3
```

주요 설정값은 다음과 같습니다.

- **`pred_steps`**: 예측 horizon입니다. 기본값은 `12`입니다.
- **`delay`**: 라벨 도착 지연입니다. `"pred_steps"`로 두면 예측 horizon과 같은 지연을 사용합니다.
- **`agg_model`**: 사용할 aggregation 알고리즘입니다. 현재 `refol`, `fedavg`, `fedostc`를 지원합니다.
- **`fedavg_client_fraction`**: FedAvg의 client fraction `C`입니다. 기본값 `1.0`은 persistent local history가 비어 있지 않은 모든 client가 참여하는 all-client FedAvg이며, `0.1`은 history 보유 client 중 10%를 seed/round 기반으로 샘플링합니다.
- **`dataset` / `adj_mx` / `num_clients`**: 데이터셋, 그래프 인접 행렬, 클라이언트 수를 설정합니다.
- **`fedostc_encoder_hidden_size` / `fedostc_decoder_hidden_size` / `period_steps`**: FedOSTC 전용 encoder hidden size, decoder hidden size, period-aware aggregation 주기를 설정합니다.

### FedAvg partial participation 동작

`fedavg`는 일반적인 FedAvg의 client-local dataset `P_k`를 delayed-label online setting에 맞춰 해석합니다.

- 모든 client는 매 round `r`의 current sample을 먼저 global model로 평가하고, 지표는 `ŷ_r`와 `y_r` 비교로 기록합니다.
- `r > delay`가 되면 각 client의 라벨 도착 sample `tau = r - delay`가 해당 client의 persistent local history `P_k`에 누적됩니다.
- `fedavg_client_fraction=C`는 history가 비어 있지 않은 client 중 local training/aggregation에 참여할 비율입니다. 선택되지 않은 client의 delayed data는 버리지 않고 local history에 보존됩니다.
- 선택된 client는 참여 round마다 자기 history 전체로 local epochs를 수행하고, server는 선택된 client의 history sample 수 `n_k`로 weighted average합니다.
- 계산량 주의: `C<1` 또는 누적 history 전체 학습은 기존 one-sample online update보다 훨씬 느릴 수 있습니다. Full-run에서는 작은 `C`, 짧은 `rounds`, 또는 별도 최적화가 필요할 수 있습니다.

### FedOSTC 구현 범위와 exactness boundary

`fedostc`는 FedOSTC 논문 경로를 이 레포의 delayed-label feedback protocol에 이식한 구현입니다. 따라서 다음 차이는 의도된 delayed-label adaptation입니다.

- **허용된 차이**: delayed scheduling을 통해 라벨 도착 이후의 feedback sample만 학습하고, current/future `y` leakage 방지를 위해 예측 시점의 미래 `y`를 입력 경로에 넣지 않습니다.
- **평가 정책**: FedOSTC도 REFOL/BaseFLServer와 동일하게 round `r`의 prediction metric을 current `y_r`로 계산해 round `r`에 기록합니다. metric 집계도 BaseFLServer처럼 client별 local log를 `num_samples`로 가중 평균합니다. 이 current `y_r` 접근은 offline benchmark evaluation 전용이며, delayed training/update/aggregation에는 `y_tau`만 사용됩니다.
- **Prediction input**: FedOSTC prediction path 자체는 계속 current speed history `x_r` only입니다. `y_r`는 model input, client selection, local OGD, aggregation에 사용하지 않습니다.
- **금지된 차이**: paper-exact 경로에 없는 non-paper covariates를 추가하지 않고, REFOL selection/KL drift 조건을 사용하지 않으며, subset participation 없이 모든 client가 delayed OGD/update 대상입니다.
- **Unresolved ambiguity**: `a(·)` projection parameterization/training, decoder `h`/`h_prime` bridge/fusion은 논문 표기가 완전히 결정적이지 않아 현재 구현의 명시적 해석입니다. Phase5의 rho source 해석은 `FEDOSTC_PHASE5_DECISION.md`를 참고하세요.

이 구현은 위 ambiguity assumption을 명시한 delayed-label adaptation이며, 정확한 저자 구현과 동일하다고 주장하지 않습니다.

시뮬레이션이 종료되면 라운드별 종합 예측 지표(RMSE, MAE)가 엑셀 파일(`.xlsx`)로 저장됩니다.

---

## ✅ 검증 (Tests)

README의 delayed-label protocol 설명과 코드 동작을 확인하는 회귀 테스트는 다음 명령으로 실행합니다.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

현재 테스트는 다음을 확인합니다.

- delayed-label scheduling에서 라벨 도착 이후에만 로컬 학습이 수행되는지
- FedOSTC가 round `r`의 prediction metric을 `y_r`로 기록하면서도 delayed OGD/update에는 `tau = r - delay`의 label만 사용하는지
- GRU 예측 경로가 `y`/`y_attr`를 읽지 않아 label leakage가 없는지
- 데이터와 metric이 raw scale을 유지하는지
- REFOL-style selection policy와 FedAvg 선택 정책이 의도대로 분리되는지

---

## 🙏 감사의 글 (Acknowledgements)

본 프레임워크의 기초 구조와 REFOL-style AttGCN aggregation 아이디어는 다음 연구와 오픈소스 코드베이스에서 영감을 받아 delayed-label Federated Online Learning 시뮬레이터로 각색 및 확장되었습니다.

- [REFOL: Resource-Efficient Federated Online Learning for Traffic Flow Forecasting](https://github.com/yuppielqx/REFOL)
- [KDD2021_CNFGNN](https://github.com/mengcz13/KDD2021_CNFGNN/tree/master)
- [DCRNN](https://github.com/liyaguang/DCRNN/blob/master)
