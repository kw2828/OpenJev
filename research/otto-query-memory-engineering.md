# Query-written memory: implementation qualification

**Status: 128 fabricated tests and lint pass. No new trained-model result.**
The earlier protected-readout experiment remains **FAIL 15/29**. These checks
qualify separate components for a new hypothesis; they do not reopen that study.

## What is implemented

The [scheduled predictor](../src/openjev/research/otto_scheduled_predictor.py)
retains the original eight tensors and 6,112 parameters. It supports actual
query periods four and eight, carries the schedule explicitly, and exposes the
hidden state and complete forecast before assimilating each later query answer.
The existing static action residual is included in that forecast.

The [memory kernel](../src/openjev/research/otto_query_memory.py) adds an explicit
episode-local score-correction matrix. It reads and writes with the same causal
key, learns only from actual later query answers, and exposes each correction
before writing. Controls include no memory, decaying last error, instantaneous
delta updates, history-conditioned delta updates, additive writes and rotated
past features. Query answers and padding retain their exact original bits.
The kernel owns no learned parameters; training a key encoder is future work.

This is **history-conditioned online residual regression**. Delta-rule fast
weights are prior art. A feature trace alone does not establish biological
learning, temporal credit assignment, a recurrent world model or novelty.
See the [design and prior-art references](otto-query-memory-design.md).

## What was checked

- Period-four outputs, priors and carry match the old predictor bitwise under
  identical weights, parameter flags and effective gradient context.
- Period-eight recurrence matches independently written GRU equations. There
  is no invented query at step four.
- Altering the current query answer cannot change its prewrite forecast;
  future information cannot change earlier outputs.
- Masked-out answers and padding can contain poisoned values without leaking
  into predictions. Consumed nonfinite operands fail visibly.
- A nonzero static-residual fixture detects use of the incomplete base prior.
- Keys receive gradients through earlier memory writes, including zero-valued
  writes. Frozen slow predictions and supplied targets receive none.
- Resets, mixed episode endings, explicit detached carry, chunking, owned state
  and hand-counted work totals satisfy their contracts.

These are small fabricated numerical and differentiation tests, not collected
episodes, model fitting or held-out evaluation. Work counts record executed
operations and coordinate terms; they are not measured speedups or hardware FLOPs.

## Qualification record

| Attempt | Tests | Lint | Outcome |
| --- | --- | --- | --- |
| [01](../output/otto-query-memory-engineering-v1/attempt-01/receipt.json) | 124 passed, 2 failed | Three import-format findings | Preserved failure |
| [02](../output/otto-query-memory-engineering-v1/attempt-02/receipt.json) | 126 passed | Pass | Projection fixture corrected |
| [03](../output/otto-query-memory-engineering-v1/attempt-03/receipt.json) | 128 passed | Pass | Added complete-prior negative control |

The first failure came from the integration fixture's key projection: a batched
matrix multiplication over the whole episode differed from the one-step tail
by at most **1.1920928955078125e-7**. Hidden keys and shadow priors were identical.
The [saved diagnostic](../output/otto-query-memory-engineering-v1/projection-diagnostic-01.json)
localizes every difference to step 64. The fixture now uses identical per-step
projection calls; its exact assertions remain. Neither production component
changed after the initial qualification attempt. Import formatting was repaired.

All three attempts retain commands, bounded process exits, logs and before/after
source hashes. Two independent source reviews found no remaining blocker.
Existing frozen predictor sources were unchanged. Reproduce the final checks:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_otto_query_memory.py tests/test_otto_scheduled_predictor.py \
  tests/test_otto_query_memory_integration.py
```

## Next evidence required

A future composed trainer must explicitly bind the slow and fast carry positions,
key-projection arithmetic, gradient boundaries, optimizer, training budget and
all comparison arms. The components do not enforce these cross-module choices.
Register fresh TRAIN/DEV/TEST cases before execution; keep the stronger ordinary
Joint AUX control and simple last-error correction. The period-eight comparison
must use schedule-specific scopes and clocks. Fixed-path forecasting cannot
establish autonomous behavior or actual teacher-call savings.

No performance advantage, model promotion, connectome result or ICLR contribution
follows from this engineering pass.
