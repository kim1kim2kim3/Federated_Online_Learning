# FedOSTC Phase1 Decision: Client Decoder Fusion

## Scope

Phase1 implements only the client-side FedOSTC encoder/decoder model. It does not implement the server GAT spatial update, client/server runtime, registry integration, delayed OGD loop, or period-aware aggregation.

## Evidence status

Checked source surfaces recorded in Phase0/Phase1 planning:

- `fedOSTC.pdf` / arXiv paper: Algorithm 2 passes both `h` and `h_prime` into `ClientDecode`, but the exact fusion bridge is not specified.
- Papers With Code entry: no official implementation surface was identified during planning.
- REFOL repository: useful as the baseline framework snapshot, but not an official FedOSTC implementation.

Therefore this Phase1 decoder is not claimed to be an exact hidden bridge from the paper. It is a documented paper-ambiguity resolution.

## Decision

Use **concat-repeat** fusion for `ClientDecode(h, h_prime, ...)`:

```text
h:        [batch, node, 64]
h_prime:  [batch, node, 64]
fused = concat(h, h_prime): [batch, node, 128]
decoder input = repeat fused for pred_steps
GRU decoder hidden size = 128
FC output head -> [batch, pred_steps, node, output_dim]
```

## Rationale

- Algorithm 2 exposes both original temporal hidden state `h` and server-refined hidden state `h_prime`; concat-repeat uses both explicitly.
- The paper default encoder hidden size is `64` and decoder hidden size is `128`; concatenating the two 64-dimensional states matches the decoder input width without adding an arbitrary projection layer.
- Repeating the fused vector across the forecast horizon avoids feeding future target values into the decoder.
- No performance-improving bridge, extra optimizer, scheduler, or covariate path is introduced in Phase1.

## Exactness boundary

Do not describe this Phase1 model as a fully exact FedOSTC implementation. The correct wording is:

> Phase1 implements the FedOSTC client encoder/decoder shape under the documented concat-repeat decoder-fusion assumption. End-to-end FedOSTC still requires the Phase2 server GAT and later delayed online-learning runtime phases.

## Follow-up

- Phase2 must document the server projection function `a(·)` before implementing Eq. (8)-(10).
- Phase3+ must ensure local OGD recomputes `h` each epoch while reusing the `h_prime` produced from the first-epoch hidden collection, as recorded in Phase0.
