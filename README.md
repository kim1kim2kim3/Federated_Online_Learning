# Delayed-Label Federated Online Learning 시뮬레이션 프레임워크

본 프로젝트는 스트리밍 데이터에서 **지금 들어온 입력은 즉시 예측하고(predict-now), 라벨은 지연(delay) 이후 도착했을 때만 학습에 사용하는(learn-later)** 지연 라벨(Delayed Label / Delayed Feedback) 환경을 타겟으로 한 **Federated Online Learning(FOL)** 시뮬레이션 프레임워크입니다.

핵심 목표는 특정 REFOL 재현 코드에 머무르는 것이 아니라, REFOL/FedAvg 같은 aggregation 알고리즘을 동일한 delayed-label online protocol 위에서 공정하게 비교할 수 있도록 만드는 것입니다. 현재 코드의 REFOL 구현은 기존 baseline 방법론을 delayed-label FOL 환경에 맞게 이식한 예시이며, 서버 구조를 확장해 다른 aggregation 전략도 같은 평가 흐름에서 실험할 수 있습니다.

---

## ✨ 주요 특징 (Key Features)

1. **Delayed feedback scheduling**
   - `delay`만큼 라벨 도착을 지연시켜 현실적인 delayed feedback 스트림을 시뮬레이션합니다.
   - `default_config.yaml`의 `delay: "pred_steps"` 설정은 예측 horizon(`pred_steps`)과 같은 길이의 라벨 지연을 사용합니다.

2. **Prequential evaluation (Predict-then-Update)**
   - 각 시점의 입력은 학습보다 먼저 평가되어 온라인 환경의 `predict-now, learn-later` 순서를 보존합니다.
   - 라벨이 도착한 과거 샘플만 학습에 투입하므로 시간 순서 기반 평가 지표(RMSE, MAE)를 계산할 수 있습니다.

3. **Label leakage 방지**
   - 현재 또는 미래 라벨을 예측 입력으로 사용하지 않도록 delayed-label scheduling과 GRU 입력 경로를 분리합니다.
   - 테스트는 예측 결과가 `y`/`y_attr` 변조에 영향을 받지 않는지 확인합니다.

4. **REFOL-style drift-based selective training**
   - 입력 피처 분포 변화(KL divergence)를 이용해 concept drift를 감지하고, 변화가 큰 클라이언트를 선택적으로 학습시킵니다.
   - 지연 라벨 상황에서도 계산/통신 비용을 줄이는 resource-efficient FOL 실험을 지원합니다.

5. **확장 가능한 FedAvg/REFOL 서버 구조**
   - 공통 delayed-label 실행 흐름은 `BaseFLServer`가 담당합니다.
   - `Refol.py`, `Fedavg.py`, `template.py`를 기반으로 aggregation 로직만 교체해 새로운 서버를 추가할 수 있습니다.

---

## 📂 폴더 구조 (Folder Structure)

```directory
Federated_Online_Learning/
├── config.py                   # 실행 인자 및 yaml 설정 로더
├── default_config.yaml         # pred_steps, delay, agg_model 등 기본 설정
├── run.py                      # 학습 프로세스 진입점
├── client_oa.py                # 로컬 클라이언트 객체 및 선택적 학습 로직
├── fl-server/
│   ├── base_server.py          # delayed-label FOL 공통 베이스 서버
│   ├── Refol.py                # REFOL-style aggregation 구현체
│   ├── Fedavg.py               # Federated Averaging(FedAvg) 구현체
│   └── template.py             # 신규 aggregation 서버 확장 템플릿
├── models/
│   ├── fl_model.py             # 클라이언트 GRU 예측 모델
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

# FedAvg delayed-label FOL 실행
.venv/bin/python run.py --agg_model fedavg
```

주요 설정값은 다음과 같습니다.

- **`pred_steps`**: 예측 horizon입니다. 기본값은 `12`입니다.
- **`delay`**: 라벨 도착 지연입니다. `"pred_steps"`로 두면 예측 horizon과 같은 지연을 사용합니다.
- **`agg_model`**: 사용할 aggregation 알고리즘입니다. 현재 `refol`, `fedavg`를 지원합니다.
- **`dataset` / `adj_mx` / `num_clients`**: 데이터셋, 그래프 인접 행렬, 클라이언트 수를 설정합니다.

시뮬레이션이 종료되면 라운드별 종합 예측 지표(RMSE, MAE)가 엑셀 파일(`.xlsx`)로 저장됩니다.

---

## ✅ 검증 (Tests)

README의 delayed-label protocol 설명과 코드 동작을 확인하는 회귀 테스트는 다음 명령으로 실행합니다.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

현재 테스트는 다음을 확인합니다.

- delayed-label scheduling에서 라벨 도착 이후에만 로컬 학습이 수행되는지
- GRU 예측 경로가 `y`/`y_attr`를 읽지 않아 label leakage가 없는지
- 데이터와 metric이 raw scale을 유지하는지
- REFOL-style selection policy와 FedAvg 선택 정책이 의도대로 분리되는지

---

## 🙏 감사의 글 (Acknowledgements)

본 프레임워크의 기초 구조와 REFOL-style AttGCN aggregation 아이디어는 다음 연구와 오픈소스 코드베이스에서 영감을 받아 delayed-label Federated Online Learning 시뮬레이터로 각색 및 확장되었습니다.

- [REFOL: Resource-Efficient Federated Online Learning for Traffic Flow Forecasting](https://github.com/yuppielqx/REFOL)
- [KDD2021_CNFGNN](https://github.com/mengcz13/KDD2021_CNFGNN/tree/master)
- [DCRNN](https://github.com/liyaguang/DCRNN/blob/master)
