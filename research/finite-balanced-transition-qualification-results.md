# Balanced transition primitive: 42 numerical checks pass

**The finite Sinkhorn primitive passes all 42 fabricated-input checks and
lint.** Its original supervised process completed in **2.699 seconds**, within
the registered 90-second cap. This qualifies a numerical component for the
[proposed transition comparison](balanced-transport-next.md); it does not
establish a trained model or a decision-performance improvement.

The [implementation](../src/openjev/research/finite_balanced_transition.py)
accepts four positive-transition logit matrices with eight states. It executes
exactly 64 row/column normalization sweeps in the log domain, then requires
strict positive probabilities and row/column residuals no greater than 1e-12.
It retains the full gradient graph. Numerical failures stop the call without
extra iterations, clipping, repairs or fallback.

| Check family | Independent evidence |
| --- | --- |
| Closed forms | Uniform and rank-one inputs give uniform transport; a positive permutation mixture is a fixed point. |
| General forward values | A separate NumPy implementation agrees on a fabricated asymmetric matrix. |
| Gradients | The uniform-input Jacobian matches analytic double-centering; asymmetric directional derivatives match central finite differences of the finite computation. |
| Invariances and ownership | Action/row/column relabeling and additive logit offsets preserve the expected result; outputs own storage and inputs remain unchanged. |
| Failure paths | Invalid domains, underflow, saturation, insufficient sweeps and callback exceptions are rejected without repair or extra iterations. |
| Work accounting | A default call performs 128 batched logsumexp reductions, 4,096 vector normalizations and 32,768 entry normalizations; counters remain available after failure. |

The work counters cover named forward operations only. They are not total
FLOPs or backward costs. The 2.699-second qualification time includes imports,
lint, all tests and cleanup; it is not model-inference latency.

## What remains before a learning experiment

The derivative is of the executed finite sequence, not an exact infinite
Sinkhorn projection. A finite number of sweeps cannot converge to the required
tolerance for every positive matrix. A uniformly mixing transition satisfies
the balancing constraint while erasing every state contrast.

Model integration is not yet qualified. In particular, the prefix likelihood
and its log prior must use the same balanced probabilities as inference. The
existing prefix bridge directly computes column-softmax log probabilities
and cannot be reused unchanged. Hazards, emissions and the cost head stay
separate; survival-weighted mass must not be balanced.

After integration checks, a new protocol must compare persistent balancing
with both original free transitions and a free control initialized to the
same effective probabilities. That trial needs fresh seeds and histories,
matched training-time allowances including normalization, complete update
accounting, and decision-based acceptance criteria. No learning trial or
closed longer-horizon-supervision follow-up is admitted by this qualification.

## Evidence

All ten registered source files remain unchanged. The complete source
snapshot, original commands, logs, worker receipt and native supervisor
launch/closure are retained. The process group exited cleanly without timeout
or termination signals. No checkpoints were opened, no model was trained and
no empirical cases were evaluated.

[Protocol](finite-balanced-transition-qualification-protocol.md) ·
[Summary](finite-balanced-transition-qualification-results/summary.json) ·
[Evidence manifest](finite-balanced-transition-qualification-results/manifest.json) ·
[Complete qualification archive](finite-balanced-transition-qualification-results/evidence.tar.gz) ·
[Publication receipt](finite-balanced-transition-qualification-results/receipt.json).

Registration SHA256:
`afc981803d96830c4946c5f9e0702c42ae0f888c1cd9be59311d9338e0905bd2`.
