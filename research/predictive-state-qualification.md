# Predictive-state correction qualification

Historical qualification at commit `1991c965`. The later
[measured multivariate comparison](fsm-correction-results.md) completes all fits
but fails its continuation rule. The original qualification record follows.

**139 fabricated tests passed. No new empirical performance result.** The core
and data adapter are implemented, but the proposed single-output benchmark
cannot test adaptive routing under the implemented equation. This limitation
was caught before measured-data access or training.

| Component | Tests | What they check |
|---|---:|---|
| Recurrent correction core | 78 | Independent equations, gradients, causal rollout, equal initialization, correction masks, local descent, fixed SISO/rank-one routing and a direction-dependent multivariate example |
| ParWH data adapter | 61 | Native chronology, complete phase groups, FIT-only normalization, input timing, immutable arrays, MAT variable whitelist and separation of forecast targets |

Both focused lint checks pass. The original core qualification process exited
successfully in 1.932 seconds; the adapter process in 1.182 seconds. These are test
process durations, **not inference latency measurements**. One earlier adapter
lint failure concerned import order and was corrected; its original source and
tool-result summary are retained. No pytest failure was hidden or retried.

## Implemented and ruled out

The model advances three state blocks through a GRU and applies decoder-derived
observation corrections to all blocks, the strongest-gradient block, or its
cyclic successor. The three variants share initial weights and training-only
future-prediction heads. Tested SISO capacity configurations contain 27,844 and
76,772 total parameters, including those heads. They are constructor checks,
not trained checkpoints or selected empirical model sizes.

For one output, every block score equals the same scalar squared residual times
that block's decoder norm. Therefore the chosen block is constant at fixed
weights whenever the residual is nonzero. It cannot establish adaptive memory
correction on ParWH. Rank-one multivariate decoders can suffer the same issue.
The [design note](predictive-state-routing-design.md) includes the derivation,
prior art and required controls for a future multivariate comparison.

The [ParWH adapter](parwh-data-contract.md) remains a reusable identification
component. Its correctness tests do not validate a trained model or establish
a novel architecture, connectome advantage, calibrated posterior or global
recurrent stability. The prior joint-observer failure remains unchanged.

## Reproduce and inspect

These focused tests require Python, NumPy, PyTorch, SciPy and pytest. The local
qualification used Python 3.12.13, NumPy 2.5.3, PyTorch 2.14.0, SciPy 1.18.1 and
pytest 9.1.1. From the repository root:

```bash
uv run --extra train --extra dev --with scipy python -m pytest --noconftest -q \
  tests/test_predictive_state_correction.py tests/test_parwh_data.py
```

[Evidence manifest](predictive-state-qualification/manifest.json) maps each
preserved file to its original repository-relative path and SHA-256. Receipts
retain the original command/log paths; the manifest locates their published
copies. It includes source snapshots, raw qualification logs, the earlier lint
failure summary and source-retrieval metadata. Source retrieval was limited to
metadata and code; no measurement archive was fetched.
