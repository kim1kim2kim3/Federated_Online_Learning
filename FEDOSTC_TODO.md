# FedOSTC 정확 구현 + Delayed Label 이식 TODO

이 문서는 `fedOSTC.pdf`의 FedOSTC를 **논문 알고리즘에 최대한 충실하게 구현**하되, 현재 레포의 delayed-label Federated Online Learning 환경 때문에 필요한 최소 수정만 적용하기 위한 작업 목록이다.

## 구현 원칙

### 핵심 원칙

> 목표는 "새로운 FedOSTC 변형"이 아니라 **논문 FedOSTC 구현에 delayed label을 붙인 버전**이다.

따라서 다음을 지켜야 한다.

- FedOSTC의 구조, 수식, all-client participation, GAT spatial update, OGD update, period-aware aggregation은 논문을 따른다.
- delayed-label 환경 때문에 불가피한 부분만 바꾼다.
- label leakage를 막기 위한 decoder target 입력 수정은 허용한다.
- 그 외 성능 개선, 단순화, 다른 알고리즘식 선택 정책, 임의 heuristic은 추가하지 않는다.
- 논문에 명시되지 않은 구현 자유도는 **unresolved paper ambiguity**로 표시하고, 임의 선택을 "정확한 FedOSTC"라고 부르지 않는다.

### 허용되는 변경

아래 두 종류만 알고리즘 변경으로 허용한다.

1. **Delayed-label scheduling**
   - 논문 FedOSTC는 round `t`에서 즉시 true future sequence `X^F_{t,n}`를 볼 수 있다고 가정한다.
   - 현재 프레임워크에서는 label이 `delay` 이후 도착하므로, local OGD update는 label이 도착한 delayed sample에 대해서만 수행한다.
   - 이 범위에는 `tau = r - delay` index mapping, label 도착 전 train/aggregation 금지, current prediction 이후 feedback 소비, end-of-stream feedback flush가 포함된다.
   - 이 범위에 포함되는 정책은 모두 **delayed-label adaptation assumption**으로 표시한다.

2. **Label leakage 방지**
   - 논문은 true future sequence `X^F`를 prediction input이 아니라 loss 계산에 사용한다.
   - 따라서 현재 프레임워크의 `y` tensor는 prediction에 영향을 주면 안 된다.
   - FedOSTC decoder는 future target `y`를 입력으로 받으면 안 된다.
   - 기존 프레임워크 API상 target-shaped tensor가 필요한 경우에만 zero/non-target placeholder를 **shape/interface shim**으로 사용한다.
   - 이 변경은 leakage 방지를 위한 최소 수정이며, future covariate 추가를 의미하지 않는다.

### 금지되는 변경

- REFOL식 drift selection 추가 금지.
- FedAvg와 비교하기 위한 임의 구조 변경 금지.
- client subset sampling 추가 금지.
- GAT를 논문 수식과 다른 PyG `GATConv` 구조로 임의 대체 금지.
- period-aware aggregation을 moving average, exponential smoothing 등으로 대체 금지.
- server가 raw `x`, raw `y`, future target value를 보는 구조 금지.
- 논문에 없는 `x_attr`, `y_attr` 또는 기타 covariate를 FedOSTC exact path에 추가 금지.
- 성능 향상을 위한 extra optimizer, scheduler, normalization, clipping 등을 논문 근거 없이 추가 금지.
- paper ambiguity를 임의로 확정하고 "논문 그대로"라고 주장 금지.

## 논문 FedOSTC 구성요소

### 1. Client-side GRU encoder

각 client `s_n`은 historical traffic speed sequence를 GRU encoder에 넣어 temporal pattern을 얻는다.

```text
X^T_{t,n} -> GRU encoder -> h_{t,n}
```

구현 요구사항:

- encoder는 client model parameter `w_{t,n,e}`에 포함된다.
- 논문 실험 설정 기본값은 encoder GRU hidden size `64`이다.
- input은 논문 Eq. (1)과 Algorithm 2의 traffic speed sequence `X^T_{t,n}`이다.
- 현재 레포 데이터의 `x_attr`는 FedOSTC exact path에서 사용하지 않는다.
- hidden temporal pattern `h_{t,n}`만 서버로 전송한다.
- 서버는 raw sequence를 받지 않는다.

### 2. Server-side GAT spatial correlation

서버는 client hidden states `{h_{t,n}}`를 받아 논문 수식 (8)-(10)에 따라 spatially updated hidden state `{h'_{t,n}}`를 계산한다.

```text
xi^t_{m,n} = a([h_{t,n} || h_{t,m}]),  for s_m in A_n
alpha^t_{m,n} = softmax_m(LeakyReLU(xi^t_{m,n}))
h'^t_n = sigmoid(sum_{s_m in A_n} alpha^t_{m,n} h^t_m)
```

구현 요구사항:

- adjacency는 기존 `data["edge_index"]` 또는 `data["adj_mx"]`를 사용한다.
- `A_n`에는 논문 정의대로 self node가 포함되어야 한다.
- attention softmax는 각 target node `n`의 neighbor set `A_n` 안에서 수행한다.
- `a(·)`는 Eq. (8)의 projection function이다.
- PyG `GATConv`로 대체하지 않는다. `GATConv`는 일반적으로 feature transform과 trainable attention을 포함하므로 논문 수식과 1:1 대응이 아니다.

#### `a(·)` paper ambiguity

논문은 `a(·)`를 projection function이라고만 정의하고, parameterization, initialization, optimizer, training rule을 명시하지 않는다.

따라서 TODO 기준은 다음과 같다.

- [ ] 구현 전 `a(·)` 처리 방침을 별도 결정 기록으로 남긴다.
- [ ] 논문/보조자료/저자 코드에서 확인되지 않은 random projection, trainable server optimizer 등을 FedOSTC exact path에 넣지 않는다.
- [ ] `a(·)`의 parameterization/training rule은 fedOSTC.pdf에서 unresolved이다. 저자 코드나 보조자료로 확인되지 않으면 random/trainable/tuned projection을 exact FedOSTC라고 부르지 않는다.
- [ ] `a(·)` 구현이 `Linear(2 * hidden_dim, 1)` 등 특정 parameterization을 필요로 하면, 이는 paper ambiguity resolution으로 표시하고 exact claim에서 분리한다.
- [ ] `a(·)`를 parameterized module로 구현하는 경우, 해당 선택은 **paper ambiguity assumption**으로 문서화한다.
- [ ] server GAT parameter는 client local OGD, client state_dict aggregation, period-aware aggregation 대상에 포함하지 않는다.
- [ ] Eq. (9)의 LeakyReLU slope는 config/tuning knob로 추가하지 않는다. paper가 값을 명시하지 않으므로 구현 기본값은 고정하고 문서화한다.

### 3. Client-side GRU decoder

서버가 `{h'_{t,n}}`를 client로 돌려주면 client는 decoder로 future sequence를 예측한다.

논문 Algorithm 2의 `ClientDecode`는 original hidden pattern과 server-updated hidden pattern을 모두 받는다. 다만 본문 설명은 decoder input으로 `h'`를 강조하므로, `h`/`h'` fusion 방식은 **paper ambiguity**로 기록한다.

```text
ClientDecode(h_{t,n}, h'_{t,n}, w_{t,n,d}) -> X_hat^F_{t,n}
```

구현 요구사항:

- decoder는 client model parameter `w_{t,n,d}`에 포함된다.
- 논문 실험 설정 기본값은 decoder GRU hidden size `128`이다.
- decoder signature는 `decode(data, h, h_prime)`처럼 original hidden `h`와 updated hidden `h_prime`을 모두 명시해야 한다.
- `decode(data, h_prime)`처럼 `h`를 생략한 단순화는 금지한다.
- 논문 본문처럼 decoder는 GRU 뒤에 final decision용 fully-connected layer를 포함해야 한다.
- encoder hidden size `64`와 decoder hidden size `128`의 연결 방식은 구현 전 명시한다.
  - 논문/보조자료에서 확인된 방식이 있으면 그대로 사용한다.
  - 확인되지 않으면 dimension bridge는 **paper ambiguity assumption**으로 기록한다.
  - dimension bridge가 필요하더라도 성능 개선용 layer를 추가하지 않는다.
- output shape는 기존 프레임워크와 동일해야 한다.

```text
[batch, pred_steps, node, output_dim]
```

- delayed-label prediction 시점에는 true future value `y`를 decoder input으로 쓰지 않는다. FedOSTC exact path의 model calculation에는 `y`도 `zeros_like(y)`도 들어가지 않아야 한다.
- 현재 레포 데이터의 `y_attr`는 FedOSTC exact path에서 사용하지 않는다.

### 4. OGD local update

논문 수식 (11)-(12)를 따른다.

```text
g^{(e)}_{t,n} = grad l(X^F_{t,n}, X_hat^F_{t,n}; w^{(e-1)}_{t,n})
w^{(e)}_{t,n} = w^{(e-1)}_{t,n} - eta * g^{(e)}_{t,n}
```

구현 요구사항:

- optimizer wrapper를 쓰더라도 의미는 OGD/SGD one-step update여야 한다.
- 기본 learning rate는 논문과 현재 설정 모두 `0.001`을 사용한다.
- 기본 local epoch는 `E=5`이다.
- local update 대상은 encoder + decoder parameter이다.
- server GAT parameter는 local OGD 대상이 아니다.
- encoder가 OGD gradient를 받지 못하는 detached hidden-only training은 금지한다.

#### Algorithm 2 local epoch detail

논문 Algorithm 2는 local epoch loop 안에서 encode/decode/loss/update를 수행하고, server GAT update `h'` 계산은 `e = 1`일 때 수행한다.

구현 요구사항:

```text
for e = 1..E:
    # gradient/loss are evaluated at w^{(e-1)}, then produce w^{(e)}
    h_{t,n}^{(e-1)} = ClientEncode(X^T_{t,n}, w_{t,n,e}^{(e-1)}, n)
    if e == 1:
        h'_{t,n} = ServerGAT({h_{t,n}^{(0)}})
    X_hat^F_{t,n} = ClientDecode(h_{t,n}^{(e-1)}, h'_{t,n}, w_{t,n,d}^{(e-1)}, n)
    loss = l(X^F_{t,n}, X_hat^F_{t,n}; w_{t,n}^{(e-1)})
    w_{t,n}^{(e)} = w_{t,n}^{(e-1)} - eta * grad(loss)
```

- `h'`는 Algorithm 2처럼 `e=1`에서 계산된 값을 local epochs 동안 재사용한다.
- 동시에 `h_{t,n}^{(e)}`는 매 epoch의 encoder parameter로 다시 계산되어 decoder에 들어가야 한다.
- 이 구조를 통해 encoder parameter도 local loss에 의해 업데이트될 수 있어야 한다.

### 5. Period-aware aggregation

논문 수식 (13)-(15)를 따른다.

초기 period 또는 rho history가 없는 구간:

```text
w_{t+1} = (1/N) * sum_n w_{t+1,n}
```

rho 계산:

```text
rho_{t,n} = exp(-||w_{t,n} - w_{t+1}||)
             / sum_m exp(-||w_{t,m} - w_{t+1}||)
```

period-aware aggregation:

```text
w_{t+1} = sum_n rho_{t+1-period,n} * w_{t+1,n}
```

구현 요구사항:

- `period_steps` 기본값은 5분 간격 traffic data 기준 `288`이다.
- 논문에서 historical length와 period가 모두 `T`로 표기될 수 있으므로 코드 변수는 반드시 분리한다.

```text
input_steps     # historical input length, 현재 데이터는 12-step x window
pred_steps      # forecasting horizon F
period_steps    # traffic period, default 288
```

- state_dict distance는 floating tensor만 대상으로 계산한다.
- non-floating tensor는 aggregation 시 dtype-safe하게 보존한다.
- numerical stability를 위해 rho는 `softmax(-distance)`로 계산해도 된다. 이는 수식과 동치다.
- 모든 client의 local state가 있어야 한다. 일부 client가 빠지는 subset aggregation은 FedOSTC exact 경로가 아니다.

## Delayed-label 이식 방식

FedOSTC 원 논문은 즉시 feedback online setting이다.

```text
논문 round t:
  receive w_t
  predict with X^T_{t,n}
  compute loss with X^F_{t,n}
  update local model immediately from w_t
  aggregate local models into w_{t+1}
```

현재 프레임워크에서는 label이 늦게 도착하므로 다음처럼 바꾼다.

```text
stream round r:
  current sample index = r
  delayed feedback sample index tau = r - delay
```

### Delayed update model policy

Delayed-label에서는 sample `tau`의 label이 stream round `r = tau + delay`에 도착한다. 이때 OGD를 어떤 model에서 시작할지는 논문이 직접 다루지 않는다.

이 TODO의 기준은 FedOSTC의 sequential update recurrence를 보존하는 것이다. 따라서 delayed training for sample `tau`는 **feedback samples `< tau`가 이미 처리되어 반영된 현재 global model**에서 시작한다. 즉, label 도착 시점의 FedOSTC update model을 `w_update[tau]`로 두고, 이 모델에서 논문 `ClientExecute(w_t, n, t)`를 sample `tau`에 대해 실행한다.

중요 구분:

```text
w_pred[r]      # sample r prediction에 실제 사용된 model; logging/evaluation provenance only
w_update[tau]  # sample tau label 도착 시 OGD를 시작하는 current global model
```

처리 원칙:

- prediction-time snapshot `w_pred[tau]`는 delayed OGD 초기값으로 사용하지 않는다.
- stored prediction snapshot으로 되돌아가 최신 delayed global sequence를 덮어쓰는 경로는 금지한다.
- delayed feedback은 `tau` 증가 순서로 하나의 global model sequence에 적용한다.
- replay/merge 정책이 필요해지면 이는 FedOSTC 원문 기능이 아니라 **delayed-label adaptation assumption**으로 별도 기록한다.

이유:

- 논문 Algorithm 2는 `t=1..R` 순서로 `w_t -> w_{t+1}` recurrence를 만든다.
- delayed-label에서는 update가 늦게 일어날 뿐, sample index `tau` 순서의 recurrence는 보존해야 한다.
- prediction-time `w_pred[tau]`에서 OGD를 시작하면 이미 처리된 feedback `< tau`의 update를 무시할 수 있어 FedOSTC sequential semantics를 더 크게 바꾼다.

### Round `r` 전체 흐름

```text
1. 현재 sample r에 대해 predict/log only
   - current global model을 client에 로드
   - `w_pred[r]`는 logging/evaluation provenance 용도로만 기록할 수 있음
   - client encodes X^T_{r,n}
   - server computes h'_{r,n} by Eq. (8)-(10)
   - client decodes prediction X_hat^F_{r,n}
   - current label `y_r`는 benchmark evaluation metric 계산에만 사용한다.
   - round `r` row에는 prediction X_hat^F_{r,n}와 current `y_r`를 비교한 RMSE/MAE를 기록한다.
   - `y_r`는 model input, local OGD, selection, aggregation에는 사용하지 않는다.
   - local update는 하지 않음

2. if r <= delay:
   - label feedback이 아직 없으므로 local OGD update 없음
   - aggregation 없음
   - global model 유지

3. if r > delay:
   - tau = r - delay
   - current global model `w_update[tau]`를 client에 로드한다
   - delayed sample tau에 대해 FedOSTC ClientExecute를 수행
   - client encodes X^T_{tau,n}
   - server computes h'_{tau,n} by Eq. (8)-(10) at e=1
   - client decodes X_hat^F_{tau,n}
   - now-arrived label X^F_{tau,n}로 OGD update 수행
   - 모든 client local model을 period-aware aggregation
   - resulting global은 delayed feedback을 반영한 최신 global로 저장한다
```

### FedOSTC update round index

period-aware aggregation의 index는 current stream round `r`가 아니라 **feedback sample index `tau = r - delay`**를 기준으로 둔다.

이유:

- 논문 FedOSTC의 `t`는 traffic sample index이자 online update index이다.
- delayed-label 환경에서는 sample `tau`의 label이 round `r`에 도착한다.
- 따라서 논문 수식의 `t`에 대응되는 것은 `tau`이다.

구현 요구사항:

```text
if tau <= period_steps:
    use Eq. (13) average aggregation
else:
    use Eq. (15) period-aware aggregation with rho[tau + 1 - period_steps]

then compute and store rho[tau]
```

### End-of-stream delayed feedback flush

논문은 모든 `t = 1..R`에 대해 update하고 최종 `w_{R+1}`을 반환한다. Delayed-label stream을 그대로 종료하면 마지막 `delay`개 sample의 label feedback이 학습되지 않을 수 있다.

구현 요구사항:

- [x] flush는 논문 원문 기능이 아니라 delayed-feedback adaptation임을 명시한다.
- [x] evaluation stream이 끝난 뒤 update-only flush phase를 둔다.
- [x] flush phase는 새로운 current evaluation metric을 만들지 않는다.
- [x] flush phase는 남은 sample `tau`의 label feedback에 대해 delayed OGD + aggregation만 수행한다.
- [x] 최종 global model은 flush까지 반영한 모델이어야 한다.
- [x] metric report에는 online prediction/evaluation rounds와 flush update rounds를 구분한다.

## 구현 TODO

### Phase 0 — 문헌/수식 mapping 고정

Status: **완료** — `FEDOSTC_PHASE0_MAPPING.md`와 `tests/test_fedostc_phase0_contract.py`가 Algorithm 2 / Eq. (8)-(15) delayed-label 매핑 및 금지 경로 source guard를 고정한다.

- [x] `fedOSTC.pdf` Algorithm 2를 코드 주석용 pseudo-code로 재정리한다.
- [x] Eq. (8)-(10), Eq. (11)-(12), Eq. (13)-(15)를 구현 함수와 1:1로 mapping한다.
- [x] delayed-label 때문에 달라지는 줄만 명시한다.
- [x] `r`과 `tau`는 1-based protocol index로 다룬다. current tensor index는 `r - 1`, delayed tensor index는 `tau - 1 = r - delay - 1`이다.
- [x] `w_pred[r]` 또는 `snapshot[r]`는 sample `r` 예측에 실제 사용한 모델로 명명하고, delay > 0일 때 논문 immediate-feedback `w_r`와 동일하다고 암시하지 않는다.
- [x] FedOSTC exact path는 `BaseFLServer.local_execute()` 또는 `client_oa.Client.local_execute()`를 호출하지 않는다. FedOSTC 전용 all-client encode -> server GAT -> decode/predict -> delayed OGD -> tau-indexed aggregation 경로를 구현한다.
- [x] current prediction/model path는 current `y` tensor를 placeholder, shape inference, branch condition, training, aggregation 어디에도 요구하지 않는다. 단, A 평가 방식에 따라 prediction `ŷ_r` 생성 이후의 offline evaluation helper는 round `r` metric 계산 전용으로 current `y_r`를 읽을 수 있다.
- [x] decoder placeholder가 필요한 경우 batch size, `pred_steps`, node count, output dim으로 생성한다.
- [x] adjacency 방향을 논문 neighbor set `A_n`에 맞게 문서화한다: edge `(m -> n)`이면 `m in A_n`로 attention denominator를 target `n` 기준으로 normalize한다.
- [x] local output의 node 차원은 single-client tensor shim인지 명시한다.
- [x] `a(·)` parameterization/training ambiguity를 별도 결정 기록으로 남긴다.
- [x] decoder가 `h`와 `h'`를 함께 받는 구조를 구현 설계에 고정한다.
- [x] delayed update model policy: delayed update는 prediction-time snapshot이 아니라 feedback `< tau`가 반영된 current global `w_update[tau]`에서 시작하도록 문서화한다.
- [x] FedOSTC-specific 구현에 REFOL drift selection이 들어가지 않도록 금지 테스트를 만든다.

### Phase 1 — 모델 구현

Status: **완료** — `models/fedostc_model.py`, `FEDOSTC_PHASE1_DECISION.md`, and `tests/test_fedostc_phase1_model.py` implement and lock the Phase1 client model under the documented **paper ambiguity assumption: concat-repeat** decoder fusion.

Phase1 exactness boundary: this is model-only. It is not end-to-end FedOSTC until Phase2 server GAT and later client/server delayed-online runtime phases are implemented.

- [x] `models/fedostc_model.py` 추가
  - [x] `FedOSTCClientModel`
  - [x] `encode(data)` returns `[batch, node, 64]`
  - [x] `decode(data, h, h_prime)` explicitly requires both Algorithm 2 hidden inputs
  - [x] `forward(data, h=None, h_prime=None)` raises unless both `h` and `h_prime` are supplied; no paper-exact prediction is attempted without server-updated hidden states
- [x] encoder GRU 구현
  - [x] 기본 hidden size `64`
  - [x] input은 historical traffic speed `x` only
  - [x] `x_attr` 사용 금지
- [x] decoder GRU 구현
  - [x] 기본 hidden size `128`
  - [x] input은 논문 Algorithm 2의 `h`, `h'`, decoder parameter에 대응
  - [x] decoder fusion은 **paper ambiguity assumption: concat-repeat**으로 결정하고 `FEDOSTC_PHASE1_DECISION.md`에 기록
  - [x] future target `y`를 decoder/model calculation에 넣지 않음
  - [x] zero placeholder가 필요한 경우에도 `y`에서 만들지 않는다; Phase1 decoder는 placeholder 자체를 사용하지 않는다
  - [x] `y_attr` 사용 금지
  - [x] encoder hidden `64` + spatial hidden `64` = decoder input `128`; 추가 bridge layer 없음
- [x] output shape 검증
  - [x] `[batch, pred_steps, node, output_dim]`
- [x] 기존 `models/fl_model.py`의 GRU를 임의로 대체하지 않는다.
- [x] source guard: Phase1 implementation file에는 REFOL selection, PyG GAT replacement, non-paper covariate path 문자열이 없어야 한다.

### Phase 2 — Server GAT 구현

Status: **완료** — `models/fedostc_gat.py`, `FEDOSTC_PHASE2_DECISION.md`, and `tests/test_fedostc_phase2_gat.py` implement and lock Eq. (8)-(10) over hidden states only.

Phase2 exactness boundary: Eq. (8)의 `a(·)` parameterization/training rule은 논문에서 unresolved이므로, 구현은 **paper ambiguity assumption: fixed mean projection**을 사용한다. 이 선택은 exact author implementation이라고 주장하지 않는다.

- [x] `models/fedostc_gat.py` 추가
- [x] Eq. (8) projection `a([h_n || h_m])` 구현
  - [x] 구현 방식은 `FEDOSTC_PHASE2_DECISION.md` paper ambiguity decision record에 연결
  - [x] random/trainable projection을 exact FedOSTC로 무표기 사용 금지
- [x] Eq. (9) neighbor-wise softmax 구현
  - [x] LeakyReLU slope는 config로 노출하지 않음
  - [x] 구현 기본값 `0.2`는 고정하고 문서화
- [x] Eq. (10) sigmoid weighted hidden update 구현
- [x] self-loop 포함 보장
- [x] edge 없는 node fallback은 self-loop만 사용
- [x] GAT input이 hidden state뿐인지 테스트
- [x] raw `x`, raw `y`, `x_attr`, `y_attr`가 server GAT로 전달되지 않는지 source guard로 테스트
- [x] server GAT parameter가 client model state_dict나 aggregation에 섞이지 않는지 empty `parameters()` / `state_dict()` 테스트

### Phase 3 — FedOSTC client 구현

Status: **완료** — `client_fedostc.py` and `tests/test_fedostc_phase3_client.py` implement the FedOSTC client-only prerequisite.

Phase3 exactness boundary: this is client-only and is not end-to-end runnable until Phase4 server runtime wires all-client scheduling, server GAT calls, delayed feedback availability, and aggregation. The client does fail fast if delayed labels or the required `update_model_state` are missing.

- [x] `client_fedostc.py` 추가
- [x] `encode_current()` 구현
- [x] `evaluate_current(h, h_prime)` 구현
- [x] `encode_delayed()` 구현
- [x] `train_delayed(h, h_prime, update_model_state)` 구현
- [x] 학습 gate prerequisite 구현

```text
train only if:
  feedback sample tau exists
  all clients have delayed feedback for tau
  current global update model w_update[tau] exists
```

- [x] current evaluation은 delayed training과 분리된 public method로 제공되어 server가 항상 먼저 호출할 수 있다.
- [x] current prediction path는 current `y`를 읽지 않고 prediction artifact만 반환한다. metric 계산은 label reveal 시점 또는 별도 offline evaluator에서 수행한다.
- [x] FedOSTC exact client는 `client_oa.Client.local_execute()`를 재사용하지 않는다.
- [x] local OGD는 encoder + decoder parameter에만 적용한다.
- [x] 모든 client가 delayed label 도착 후 참여하는 policy는 Phase4 server runtime 책임으로 남기고, Phase3 client에는 REFOL식 selected flag를 쓰지 않는다.
- [x] detached `h_prime` 때문에 encoder가 gradient를 못 받는 구조인지 테스트한다.
- [x] `h`는 매 local epoch마다 current encoder parameter로 재계산한다.
- [x] `h_prime`은 Algorithm 2처럼 `e=1`에서 계산된 값을 local epochs 동안 재사용한다.

### Phase 4 — FedOSTC server 구현

Status: **완료** — `fl-server/Fedostc.py`가 Phase3 `FedOSTCClient`, Phase2 `FedOSTCGAT`, delayed-label scheduling, prediction provenance, all-client delayed OGD, flush, and tau-indexed aggregation을 연결한다.

Phase4 delayed-label boundary: current prediction path는 current `y`를 읽지 않고 먼저 수행하며, label이 reveal된 delayed sample `tau = r - delay`만 local OGD와 aggregation에 사용한다. `w_pred`는 provenance/logging only이고 train init에는 사용하지 않는다.

Phase4 evaluation boundary: FedOSTC round metric은 REFOL/BaseFLServer와 같은 기준으로 current prediction `ŷ_r`와 current label `y_r`를 비교해 round `r`에 기록한다. metric 집계도 BaseFLServer처럼 client별 local log를 `num_samples`로 가중 평균한다. 이 current `y_r` 접근은 offline benchmark evaluation 전용이며, delayed training/update/aggregation은 계속 `tau = r - delay`의 `y_tau`만 사용한다. Prediction path 자체는 `x_r` only이다.

- [x] `fl-server/Fedostc.py` 추가
- [x] `BaseFLServer` 상속
- [x] `boot()` 구현
  - [x] FedOSTC 전용 client 생성
  - [x] server GAT 생성
  - [x] rho history 초기화
  - [x] prediction provenance buffer는 선택 사항이며 train init에는 사용하지 않음
  - [x] global model 초기화 정책 명시
- [x] `train_round(r)` override
  - [x] `update_train_data(r, clients)`의 indexing은 참고할 수 있으나 `BaseFLServer.local_execute()`는 호출하지 않음
  - [x] `update_train_data(r, clients)` 호출
  - [x] all clients receive current global `w_r` before current evaluation
  - [x] current prediction에 사용한 model은 `w_pred[r]`로 logging/provenance 용도만 기록 가능
  - [x] current evaluation hidden collection
  - [x] server GAT current update
  - [x] current evaluation 실행 및 round `r` metric 반환
  - [x] `r <= delay`면 여기서 종료하고 global 유지
  - [x] `r > delay`면 delayed feedback training 실행
  - [x] delayed sample `tau = r - delay`는 current global `w_update[tau]`에서 시작
  - [x] delayed hidden collection
  - [x] server GAT delayed update at local epoch `e=1`
  - [x] all-client OGD update
  - [x] period-aware aggregation indexed by `tau`
  - [x] metric aggregation 반환
- [x] end-of-stream flush phase 구현
  - [x] 남은 delayed feedback samples에 대해 update-only 수행
  - [x] flush rounds는 prediction metric에 섞지 않음
- [x] `run.py` registry에 `fedostc` 추가

### Phase 5 — Period-aware aggregation 구현

Status: **완료** — `fl-server/Fedostc.py` implements Eq. (13), Eq. (14), Eq. (15), rho history, dtype-safe non-floating state preservation, and fail-fast all-client aggregation.

Phase5 exactness boundary: Eq. (14)의 rho source는 논문 표기가 모호하므로 `FEDOSTC_PHASE5_DECISION.md`에 **paper ambiguity assumption: post-local-update states vs fresh uniformly averaged global**로 기록했다. 이는 exact author implementation claim이 아니다.

- [x] `average_aggregate(local_states)` 구현: Eq. (13)
- [x] `compute_model_distances(local_models_for_rho, fresh_global)` 구현: Eq. (14)의 norm
- [x] `compute_rho(distances)` 구현: Eq. (14)
  - [x] Eq. (14)는 논문 표기상 `||w_{t,n} - w_{t+1}||`를 사용한다.
  - [x] `w_{t,n}`가 pre-OGD local snapshot인지 post-local-update state인지 논문/Algorithm 2 표기가 모호하므로, 구현 전 이 항목을 **paper ambiguity**로 남긴다.
  - [x] 어떤 해석을 선택하든 `rho` 계산 대상 모델을 코드/README에 명시하고, 이를 논문 exact가 아니라 ambiguity resolution으로 표시한다.
  - [x] 다음 round 최신 global `w_r` 등 Eq. (14)에 없는 모델 기준으로 rho를 계산하지 않는다.
- [x] `period_aggregate(local_states, rho)` 구현: Eq. (15)
- [x] rho history 저장

```text
rho_history[tau] = rho_for_tau
```

- [x] Eq. (14)의 `w_{t,n}`/`w_{t+1}` source state 해석을 구현 전 decision record로 남긴다.
- [x] 변수명은 `rho_source_local_states[tau]`, `fresh_global_for_rho[tau]`처럼 pre/post-update 혼동을 피한다.

- [x] off-by-one 테스트

```text
if tau <= period_steps:
    Eq. (13)
else:
    Eq. (15) with rho_history[tau + 1 - period_steps]
```

- [x] 모든 client가 참여하는 기본 경로만 구현한다.
- [x] missing delayed data가 있으면 subset aggregation하지 말고 fail-fast한다.
- [x] subset aggregation이 필요해지는 경우 FedOSTC exact path가 아니므로 별도 explicit variant로 분리한다.

### Phase 6 — Config/README 업데이트

Status: **완료** — `default_config.yaml` exposes FedOSTC defaults without non-paper options, and `README.md` documents the delayed-label adaptation boundary, forbidden differences, unresolved ambiguities, and run examples.

- [x] `default_config.yaml`
  - [x] `agg_model` 예시에 `fedostc` 추가
  - [x] `fedostc_encoder_hidden_size: 64`
  - [x] `fedostc_decoder_hidden_size: 128`
  - [x] `period_steps: 288`
  - [x] `fedostc_gat_negative_slope` 추가 금지
  - [x] `use_x_attr`/`use_y_attr` 같은 FedOSTC exact path 옵션 추가 금지
- [x] `README.md`
  - [x] FedOSTC는 논문 구현을 delayed-label feedback에 맞춰 이식한 것임을 명시
  - [x] 허용된 차이: delayed scheduling, future-y leakage 방지
  - [x] 금지된 차이: non-paper covariates, REFOL selection, subset participation
  - [x] unresolved ambiguity: `a(·)` parameterization/training, decoder hidden bridge, Phase5 rho source는 `FEDOSTC_PHASE5_DECISION.md` 참고
  - [x] 실행 예시 추가

### Phase 7 — 정확성 테스트

Status: **완료** — Phase1~6의 단위 테스트와 `tests/test_fedostc_phase4_server.py`가 A 평가 방식(`ŷ_r` vs `y_r`), REFOL/BaseFLServer-style client-local metric aggregation, delayed update(`tau = r - delay`), no-selection, speed-only, GAT, decoder, OGD, period aggregation, registry, flush contract를 고정한다.

- [x] FedOSTC no-selection test
  - [x] delayed label 도착 후 모든 client가 OGD update 대상이어야 함
  - [x] REFOL의 `selected` 또는 KL drift 조건을 사용하지 않아야 함
- [x] speed-only input test
  - [x] `x_attr`, `y_attr`가 FedOSTC model/GAT/decoder input으로 들어가지 않아야 함
- [x] delayed-label protocol test
  - [x] `r <= delay`에서도 current `ŷ_r` vs `y_r` 기준 finite RMSE/MAE를 기록하고 local train은 없음
  - [x] `r > delay`에서는 sample `tau = r - delay`만 train에 사용
  - [x] 반환 metric은 delayed `tau`가 아니라 current round `r` 기준임
  - [x] metric 집계는 REFOL/BaseFLServer처럼 client별 local log를 `num_samples`로 가중 평균함
  - [x] current sample `r`은 train에 사용되지 않음
- [x] delayed update model policy test
  - [x] delayed training은 current global `w_update[tau]`에서 시작해야 함
  - [x] prediction-time `w_pred[tau]`로 되돌아가 학습하지 않아야 함
  - [x] feedback samples는 `tau` 증가 순서로 처리되어야 함
- [x] end-of-stream flush test
  - [x] 마지막 `delay`개 sample의 delayed feedback update가 flush에서 처리되어야 함
  - [x] flush metric이 prediction metric에 섞이지 않아야 함
- [x] predict-then-update order test
  - [x] current evaluation이 delayed OGD보다 먼저 발생
- [x] leakage test
  - [x] `predict_current()`는 current `y` 없이 동작해야 함
  - [x] future `y` 변경이 current prediction을 바꾸지 않아야 함
  - [x] zero/non-target placeholder는 shape/interface filler로만 쓰여야 함
  - [x] `x_attr`/`y_attr`는 paper-exactness 때문에 쓰이지 않아야 함
- [x] update-only feedback test
  - [x] `update_from_feedback()`는 RMSE/MAE metric 없이 delayed OGD/aggregation stats만 반환해야 함
- [x] GAT equation test
  - [x] Eq. (8)-(10)과 같은 attention normalization인지 확인
  - [x] directed/asymmetric adjacency에서도 softmax가 target node별 neighbor set `A_n`에서 수행되는지 확인
  - [x] server GAT parameter가 aggregation/client optimizer 대상이 아닌지 확인
- [x] decoder hidden-state test
  - [x] `ClientDecode`가 `h`와 `h_prime`을 모두 받는지 확인
  - [x] `decode(data, h_prime)` 단순화가 없는지 확인
- [x] local epoch / OGD test
  - [x] gradient/loss는 `w^{(e-1)}`에서 계산하고 `w^{(e)}`를 생성
  - [x] `h_prime`은 `e=1`에서 계산 후 재사용
  - [x] `h`는 local epoch마다 재계산
  - [x] Adam/momentum/weight decay/scheduler/gradient clipping이 없어야 함
  - [x] 기본 `E=5`, `lr=0.001` 유지
- [x] period aggregation test
  - [x] `tau <= period_steps`에서 Eq. (13)
  - [x] `tau > period_steps`에서 Eq. (15)
  - [x] `rho_history[tau + 1 - period_steps]` lookup 검증
- [x] registry test
  - [x] `--agg_model fedostc`가 `FedOSTC` server를 선택
- [x] full suite
  - [x] `.venv/bin/python -m unittest discover -s tests -v`

### Phase 8 — Smoke run

Status: **완료** — A 평가 방식 smoke acceptance는 round 1부터 RMSE/MAE가 NaN이 아니고, delayed update/period-aware path/flush가 metric row 추가 없이 동작하는 것이다.

- [x] 최소 round 실행

```bash
.venv/bin/python run.py --agg_model fedostc --rounds 2
```

- [x] delayed update + period-aware path 실행 확인

```bash
.venv/bin/python run.py --agg_model fedostc --delay 1 --rounds 3 --period_steps 1
```

- [x] period-aware path smoke 확인
  - [x] test config에서 `period_steps`를 작게 override
  - [x] production default는 `288` 유지
- [x] flush path smoke 확인
  - [x] `--delay 1`로 final delayed feedback update가 처리되는지 확인
  - [x] flush는 writer에 추가 metric row를 만들지 않음
- [x] metric 확인
  - [x] round 1부터 RMSE/MAE가 NaN이 아님
  - [x] RMSE
  - [x] MAE
  - [x] no NaN

## 구현 리스크와 처리 원칙

### 1. `a(·)` projection function의 학습 여부

논문은 GAT attention score projection `a(·)`를 정의하지만, Algorithm 2에는 server GAT parameter를 어떻게 학습하는지 명시하지 않는다.

처리 원칙:

- 임의 server optimizer를 추가하지 않는다.
- Eq. (8)-(10)을 직접 구현한다.
- parameterization이 필요하면 `a(·)` 구현 형태를 코드와 README에 명시하고, exact claim이 아니라 paper ambiguity resolution으로 표시한다.
- 해당 선택은 paper ambiguity assumption이며, 확인 전에는 "정확한 FedOSTC와 완전히 동일"하다고 주장하지 않는다.

### 2. Delayed label과 period index

논문에서는 update가 sample index `t`와 동시에 일어난다. delayed-label에서는 update가 stream round `r`에서 일어나지만 sample은 `tau = r - delay`이다.

처리 원칙:

- FedOSTC 수식의 `t`는 delayed feedback sample index `tau`에 대응시킨다.
- period-aware aggregation도 `tau` 기준으로 index한다.
- delayed local update는 feedback `< tau`가 반영된 current global `w_update[tau]`에서 시작한다. prediction-time `w_pred[tau]`로 되돌아가는 것은 금지한다.

### 3. Early rounds

`r <= delay`에는 label feedback이 없다.

처리 원칙:

- prediction/evaluation만 수행한다.
- local OGD update 없음.
- aggregation 없음.
- global model 유지.

### 4. End-of-stream flush

stream 종료 후 마지막 delayed feedback이 남을 수 있다.

처리 원칙:

- update-only flush phase를 둔다.
- flush는 final global model에는 반영한다.
- flush는 prediction metric에는 섞지 않는다.

### 5. Missing client feedback

현재 데이터셋 구조에서는 모든 client가 같은 delay로 label을 받는다고 보는 것이 기본이다.

처리 원칙:

- core implementation은 all-client participation을 따른다.
- missing feedback이 있으면 subset aggregation하지 말고 fail-fast한다.
- subset behavior가 필요하면 FedOSTC exact가 아닌 별도 variant로 분리한다.

### 6. Non-paper covariates

논문 FedOSTC는 traffic speed sequence를 입력으로 정의한다.

처리 원칙:

- FedOSTC exact path에서는 `x_attr`, `y_attr`를 사용하지 않는다.
- covariate variant가 필요하면 별도 이름의 variant로 분리한다.

## 완료 기준

- [x] FedOSTC 구현은 논문 Algorithm 2와 Eq. (8)-(15)에 대응되는 함수/주석을 가진다.
- [x] delayed-label 때문에 변경된 지점이 문서화되어 있다.
- [x] `r <= delay`에서는 train/aggregation이 발생하지 않는다.
- [x] `r > delay`에서는 `tau = r - delay` sample로만 OGD update한다.
- [x] delayed update는 prediction-time snapshot이 아니라 current global `w_update[tau]`에서 시작한다.
- [x] stream 종료 후 delayed feedback flush가 수행된다.
- [x] current sample은 prediction/logging/evaluation only이며 current label `y_r`는 metric 계산 전용이다.
- [x] current prediction metric은 round `r`에 기록되며 학습/선택/aggregation과 분리된 offline evaluator에서만 계산된다.
- [x] future target `y`는 prediction input으로 사용되지 않으며, placeholder shape 생성에도 쓰이지 않는다.
- [x] `x_attr`, `y_attr`는 FedOSTC exact path에서 사용되지 않는다.
- [x] all-client participation이 기본이며 missing feedback은 fail-fast한다.
- [x] REFOL drift selection이 FedOSTC 경로에 들어가지 않는다.
- [x] decoder interface는 Algorithm 2처럼 `h`와 `h_prime`을 모두 받으며, fusion 방식 ambiguity가 문서화되어 있다.
- [x] local epoch loop는 Algorithm 2의 `e=1` server GAT update semantics를 보존한다.
- [x] server GAT parameter는 client optimizer/aggregation 대상이 아니다.
- [x] period-aware aggregation이 Eq. (13)-(15)를 따른다.
- [x] Eq. (14)의 `w_{t,n}` 해석을 포함한 paper ambiguity 항목은 코드/README에 명시된다.
- [x] 전체 unittest가 통과한다.
