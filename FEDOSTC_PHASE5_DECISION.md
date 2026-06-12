# FedOSTC Phase5 Aggregation Decision

Phase5 implements the minimum period-aware aggregation needed by the delayed-label
FedOSTC server runtime.

## Implemented scope

- Eq. (13): uniform average over every client state.
- Eq. (14): distance-based rho with `softmax(-distance)` for numerical stability.
- Eq. (15): period-aware aggregation using stored rho.
- Non-floating tensors are cloned from client 0 without dtype conversion.

## Eq. (14) ambiguity

The paper's notation `||w_{t,n} - w_{t+1}||` does not fully disambiguate whether
`w_{t,n}` is the pre-OGD local snapshot or the post-local-update client state.
This implementation fixes the reproducible interpretation to:

> compute rho from post-local-update client states against the freshly uniformly
> averaged global model produced from those same states.

This is a documented paper-ambiguity assumption, not a claim that the author
implementation made the same choice.

## Period index

For delayed feedback sample `tau`:

- `tau <= period_steps` uses Eq. (13).
- `tau > period_steps` uses Eq. (15) with `rho_history[tau + 1 - period_steps]`.

Missing rho or missing client states fail fast instead of falling back to subset
aggregation.
