# FedOSTC Phase0 Mapping: Algorithm 2 / Eq. (8)-(15) to Delayed-Label Framework

Phase0 scope is **specification plus contract tests only**. It fixes the paper-to-code mapping for a future FedOSTC implementation, but it intentionally does not add a `fedostc` runtime registry entry, model class, server class, or client class.

## Source anchors

- Paper artifact: `fedOSTC.pdf`.
- Required paper surface: **Algorithm 2: FedOSTC**.
- Required equation surface: **Eq. (8)-(10)** server-side spatial update, **Eq. (11)-(12)** client OGD update, and **Eq. (13)-(15)** period-aware aggregation.

## Index and delayed-label convention

The paper uses round `t` as both the online sample index and the immediate feedback/update index. The current framework reveals labels after `delay` stream rounds, so the paper round must be mapped to the feedback sample index.

```text
stream round r                 # 1-based online/prediction protocol index
feedback sample tau = r - delay # 1-based paper Algorithm 2 / Eq. index
current tensor index = r - 1
delayed tensor index = tau - 1 = r - delay - 1
```

Decision:

- Paper `t` maps to delayed feedback sample `tau = r - delay` whenever FedOSTC performs OGD or aggregation.
- Current round `r` is **predict/log/evaluate only**. The prediction path itself remains `x_r` only and must not use current `y` for placeholders, branch conditions, training, or aggregation. After prediction, the offline benchmark evaluation helper may read current `y_r` solely to compute the round `r` metric (`ŷ_r` vs `y_r`).
- Feedback round `tau` is the only sample eligible for delayed FedOSTC `ClientExecute` once round `r = tau + delay` is reached.
- Delayed feedback is processed in increasing `tau` order on one global model sequence.

## Model provenance policy

```text
w_pred[r]      # model actually used to predict sample r; provenance/evaluation logging only
w_update[tau]  # current global model after feedback samples < tau; delayed OGD starts here
```

Decision:

- `w_pred[r]` is **provenance only** and is not a training initial state.
- Delayed OGD for sample `tau` starts from `w_update[tau]`, the current global model after all feedback samples `< tau` have been applied.
- A future implementation must not restore `w_pred[tau]`, `snapshot[tau]`, or any prediction-time snapshot to initialize delayed training.

## Algorithm 2 mapping

### Paper immediate-feedback shape

```text
for t = 1..R:
    for each client s_n in S:
        w_{t+1,n} = ClientExecute(w_t, n, t)
    if t <= T:
        w_{t+1} = average_aggregate(w_{t+1,n})              # Eq. (13)
    else:
        w_{t+1} = period_aggregate(w_{t+1,n}, rho[t+1-T])   # Eq. (15)
    rho[t] = compute_rho(source_local_states, w_{t+1})      # Eq. (14)
```

### Delayed-label adaptation shape

```text
for stream round r = 1..R:
    # current sample r: prediction/logging only
    load current global model for every client
    h_r,n = ClientEncode(X^T_{r,n})
    h'_r,n = server_gat({h_r,n})                            # Eq. (8)-(10)
    yhat_r,n = ClientDecode(h_r,n, h'_r,n)
    record prediction and optional w_pred[r] provenance
    do not read current y_r,n in the prediction/model/update path
    offline evaluator may compare yhat_r,n with current y_r,n for round r metrics only

    if r <= delay:
        keep global model unchanged
        continue

    tau = r - delay
    # delayed sample tau: paper Algorithm 2 update mapped to feedback index tau
    load w_update[tau] for every client
    for each client s_n in S:
        w_{tau+1,n} = ClientExecute(w_update[tau], n, tau)
    if tau <= period_steps:
        w_{tau+1} = average_aggregate(w_{tau+1,n})           # Eq. (13)
    else:
        w_{tau+1} = period_aggregate(
            w_{tau+1,n}, rho_history[tau + 1 - period_steps]
        )                                                    # Eq. (15)
    rho_history[tau] = compute_rho(rho_source_state, w_{tau+1}) # Eq. (14)
```

End-of-stream flush, if implemented later, is a delayed-label adaptation rather than a FedOSTC paper feature. It may perform update-only delayed feedback for remaining `tau` samples, but it must not create new current prediction metrics.

Implementation bans fixed by Phase0: No `BaseFLServer.local_execute()` reuse. Do not replace exact path with PyG `GATConv`.

## Function and equation mapping

| Paper anchor | Future implementation responsibility | Exact-path constraints |
| --- | --- | --- |
| Algorithm 2 `ServerExecute` lines 2-4 | FedOSTC server all-client loop | No subset participation; no REFOL drift selection; no `BaseFLServer.local_execute()` reuse. |
| Algorithm 2 `ClientExecute(w_t,n,t)` | FedOSTC-specific all-client encode → server GAT → decode/predict → OGD path | Do not call `client_oa.Client.local_execute()`; do not train from `w_pred`/snapshot. |
| Eq. (8) `xi = a([h_n || h_m])` | server GAT attention-score function | Hidden states only; no raw `x`, raw `y`, `x_attr`, or `y_attr`; `a(·)` ambiguity remains unresolved. |
| Eq. (9) neighbor softmax | server GAT normalization by target node neighbor set `A_n` | If edge `(m -> n)`, then `m in A_n`; denominator is target `n`'s neighbor set. |
| Eq. (10) sigmoid weighted sum | server GAT hidden update | Include self node in `A_n`; do not replace exact path with PyG `GATConv`. |
| Eq. (11) gradient | client local OGD gradient at `w^{(e-1)}` | Loss uses delayed `X^F_{tau,n}` only after feedback is revealed. |
| Eq. (12) OGD step | client local SGD/OGD update | Encoder and decoder parameters update; no Adam/momentum/scheduler/clipping unless documented as non-exact variant. |
| Eq. (13) average aggregation | `average_aggregate(local_states)` | All clients required; floating tensors averaged; non-floating tensors preserved dtype-safely. |
| Eq. (14) rho calculation | `compute_rho(distances)` as `softmax(-distance)` | Rho source state is unresolved; selected source must be documented before implementation. |
| Eq. (15) period-aware aggregation | `period_aggregate(local_states, rho_history[tau + 1 - period_steps])` | Period index is `tau`, not current stream round `r`; default `period_steps = 288`. |

## ClientExecute local epoch contract

Algorithm 2 computes server GAT updated hidden states only when `e = 1`, while client hidden states are recomputed each local epoch from current encoder parameters.

```text
for e = 1..E:
    h_tau,n^(e-1) = ClientEncode(X^T_tau,n, w_tau,n,e^(e-1), n)
    if e == 1:
        h'_tau,n = ServerGAT({h_tau,n^(0)})                  # Eq. (8)-(10)
    X_hat^F_tau,n = ClientDecode(h_tau,n^(e-1), h'_tau,n, w_tau,n,d^(e-1), n)
    loss = l(X^F_tau,n, X_hat^F_tau,n; w_tau,n^(e-1))
    w_tau,n^(e) = w_tau,n^(e-1) - eta * grad(loss)           # Eq. (11)-(12)
```

Decision:

- `h_prime` is computed from the `e = 1` hidden collection and reused across local epochs, matching Algorithm 2.
- `h` is recomputed every local epoch so encoder parameters can receive gradients.
- Default local epoch is `E = 5`; default learning rate is `eta = 0.001`.

## Prediction path no-leakage contract

FedOSTC exact prediction uses historical traffic speed sequence `X^T_{r,n}` and spatially updated hidden states. It does not use future target labels.

Decision:

- Current `y` is not read during current prediction/model execution for placeholders, shape inference, branch conditions, training gates, or aggregation.
- Round metrics are the A-contract offline benchmark exception: after `ŷ_r` is produced from `x_r`, an evaluation helper may read current `y_r` only to record RMSE/MAE for round `r`.
- If an interface needs a target-shaped placeholder, create it from explicit `batch_size`, `pred_steps`, `node_count`, and `output_dim`; do not derive it from `y` or `zeros_like(y)`.
- FedOSTC exact path does not use `x_attr` or `y_attr`. Covariate variants must be separate variants, not the paper-exact path.
- Local output shape remains `[batch, pred_steps, node, output_dim]`. If a single-client node shim is needed, document it at implementation time.

## Paper ambiguity decision records

These items are unresolved in `fedOSTC.pdf`; Phase0 freezes them as explicit future decisions instead of silently choosing an implementation.

### 1. `a(·)` projection parameterization/training rule unresolved

Eq. (8) defines `a(·)` as a projection function but does not specify parameterization, initialization, optimizer, or training rule. A future `Linear(2 * hidden_dim, 1)`, random projection, fixed projection, or trainable server module is a **paper ambiguity resolution**, not an unqualified exact-paper fact.

Required before implementation:

- Record the chosen `a(·)` parameterization and training/non-training rule.
- Keep server GAT parameters out of client OGD and client state aggregation unless a separately named non-exact variant is created.
- Do not expose LeakyReLU slope as a tuning knob for the exact path; if a default is required, fix and document it.

### 2. `ClientDecode(h, h_prime, w_d, n)` h/h_prime fusion unresolved

Algorithm 2 calls `ClientDecode(h, h_prime, w_d, n)`, while surrounding prose emphasizes the updated hidden pattern `h_prime`. The fusion method between original `h` and updated `h_prime` is not specified.

Required before implementation:

- Preserve an interface that explicitly accepts both `h` and `h_prime`.
- Document the chosen fusion/bridge, especially because the paper uses encoder GRU hidden size `64` and decoder GRU hidden size `128`.
- Do not simplify the exact path to `decode(data, h_prime)` without recording a non-exact variant.

### 3. Eq. (14) rho source state unresolved

Eq. (14) names the distance between local model state `w_{t,n}` and fresh global `w_{t+1}`, but Algorithm 2 also produces post-local-update states `w_{t+1,n}`. The exact source local state used for rho distance is ambiguous.

Required before implementation:

- Record whether `rho_source_local_states[tau]` means pre-OGD client snapshots, post-local-update states, or another paper-supported interpretation.
- Name the fresh global used for rho as `fresh_global_for_rho[tau]` or equivalent.
- Do not compute rho from an unrelated latest global model, from prediction snapshots, or from selected-client subsets.
