# Bayesian memory screen: collection completed, evaluation did not start

The registered experiment ended with a technical failure during evaluation
planning. It produced no model-performance comparison. The source, algorithms,
selection rule and resource caps remain unchanged.

## What completed

The original collection supervisor exited successfully after **197.3103 seconds**.
All **18 DEV paths and 4,816 retained states** completed, with 18 resets and
4,816 teacher and native-step calls. The worker used at most **737,132,544 bytes
of RSS**. Its 15 payloads total **3,710,482 bytes**, excluding the receipt and
supervisor evidence. No new gradients were computed, confirmation was not
collected, and the previous experiment's TEST was not accessed.

- [Prospective registration](../output/otto-residual-estimator-v1/registration-01.json)
- [Collection receipt](../output/otto-residual-estimator-v1/dev-collection-01/receipt.json)
- [Original supervisor closure](../output/otto-residual-estimator-v1/dev-collection-native-01.terminal.json)
- [Evaluation planning failure](../output/otto-residual-estimator-v1/dev-evaluation-plan-failure-01.json)
- [Technical closure](../output/otto-residual-estimator-v1/dev-technical-closure-01.json)

The collection's registered array serialization check completed. The **evaluation
planner** decoded no numerical arrays or checkpoints, and no evaluation worker
was launched. These are distinct statements.

## Why evaluation failed

`run_otto_residual.authenticate_collection` calls the native collector's existing
authentication chain. That chain eventually reaches
`qualify_otto_released_native.authenticate`, which requires the current Python
executable and package inventory to match the native collection environment.
The evaluator separately requires its original training environment. The
planner therefore raised `ValueError: separate native reference interpreter`.

Changing the launch interpreter cannot satisfy both checks. The prior 531 tests
used a fabricated collection-authentication boundary, and the capacity probe
started after admission. Neither exercised the two real environments together.
Their passing results remain valid for their narrower scope; they did not
establish that this complete execution path worked.

The failed command was not replayed. Its failure record is transcribed from the
completed tool output, with the original input hashes and command; it is not
presented as a separately captured stderr stream. All 177 frozen source and
evidence files remain unchanged. The original collector checks resources on
every callback, more frequently than the protocol's 250-millisecond description
for evaluation and audit. All collection work remained within its fixed caps.

## Separate repair, before another experiment

Use a small native metadata bridge to authenticate collection evidence under
the exact native interpreter. It should save source and payload hashes, runtime
identity, and successful original collection closure without loading numerical
arrays or entering simulator/model setup. A new evaluator should authenticate
that bridge receipt and its original supervisor closure using only standard
library code, while retaining its own exact training-runtime check.

The regression must cross both real process boundaries: native bridge first,
then the evaluator's plan-only path. Deny numerical imports in both metadata
phases, and test changed input hashes, the wrong interpreter and an incomplete
bridge receipt. Preserve the original failed screen and all its bound files.
Do not patch interpreter identity or weaken either runtime requirement.

The collected paths can be retained as **prior development data** in a separately
registered study. That would be a disclosed new use, not a retry or a completed
result from this closed experiment. Its confirmation paths and the previous
study's TEST remain unopened. The repair is a proposal, not implemented or
qualified by this report.
