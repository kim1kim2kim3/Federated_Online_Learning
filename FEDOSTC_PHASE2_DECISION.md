# FedOSTC Phase2 Decision — Server GAT Projection Boundary

## Decision

Phase2 implements the server hidden-state update for Eq. (8)-(10) with a fixed,
state-free projection:

```text
a([h_n || h_m]) = mean([h_n || h_m])
```

The LeakyReLU slope for Eq. (9) is fixed at `0.2`.

## Why this is an assumption

`fedOSTC.pdf` names `a(·)` as a projection function but does not specify its
parameterization, initialization, optimizer, server-side training rule, or how
it should interact with client local OGD and later aggregation.

Because that behavior is unresolved in the paper, Phase2 does **not** add a
random projection, trainable server module, or server optimizer. The fixed mean
projection is only a reproducible ambiguity resolution needed to unlock the
Phase3 client prerequisite. It is not claimed to be the exact author
implementation of `a(·)`.

## Scope

- Applies only to `models/fedostc_gat.py`.
- Operates only on hidden states shaped `[batch, node, hidden]`.
- Uses directed edges as `source -> target`, with target-local softmax.
- Adds self-loops for every node before attention normalization.
- Keeps server GAT state out of client model state and aggregation.

## Rejected alternatives

- Trainable projection layer — rejected because the paper does not define its
  training rule or aggregation relationship.
- Random fixed projection — rejected because it adds seed-dependent behavior
  without paper support.
- Generic graph attention layer replacement — rejected because it can include
  feature transforms and learned attention terms not specified by Eq. (8)-(10).
