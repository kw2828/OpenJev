# Fresh evaluation of the six reward-residual models

Status: protocol frozen for a new evaluation-only attempt. No efficacy result
is available yet. This does not resume the
[stopped v1 experiment](reacher-reward-residual-pilot.md).

The [random-stream correction](reacher-rng-independence-correction.md) preserves
every trained model and fixes the evaluation's cross-role seed aliases. The
new [protocol](../evidence/reacher-reward-residual-control-v2/protocol/plan.json)
has SHA-256 `4c21a1505245314a923325ca0aae141daed6a737170d870272860bd00857561f`.
Implementation commit: `727d473`.

## Fixed comparison

Evaluate all six unchanged final checkpoints: free reward head and known-cost
residual head, each trained with seeds 271, 283 and 293. Training remains the
same 768 episodes, 48 epochs, initialization pairing and minibatch order.
There is no new training, checkpoint selection or model pruning.

Generate 96 common prediction episodes and 64 paired control cases under full
sensing, six-step gaps and ten-step shifted gaps. Preserve both memory-reset
diagnostics and all four references: supplied-physics control with true state,
supplied-physics control with a particle filter, zero commands and uniform
commands. Every learned planner retains the same 64 candidates and 12-step
horizon. All seventeen original useful-effect checks remain required.

The frozen manifest enumerates 692 root allocations and 820 concrete generator
identities. It checks across roles and against both old evaluation allocations
and original training. Pairing across controllers and panels is declared;
actuator disturbances never intentionally share a generator with planner
samples. Distinct seeds and initial states prevent these aliases, rather than
proving mathematical independence of pseudorandom sequences.

## Costs, stopping and verification

One 1,800-second cooperative cap covers validation, copying, model loading,
fresh data collection, prediction, control and final member hashing. Python
imports and the final completion-receipt write are outside that interval.
Failure preserves partial artifacts and the active phase. No retries, resumes,
replacement seeds or cap extensions.

Inherited fits cost 209.151973 seconds within the prior invalid attempt's
359.138964 seconds. Report new evaluation time separately. Actual cumulative
cost for this two-attempt chain is prior invalid time plus new evaluation time;
adding fitting again would double-count it. Shared-host timings are descriptive.

The new audit authenticates all copied fit files, lineage, paired initialization,
minibatch orders, expanded stream manifest and complete controller coverage.
It replays every saved native transition and recomputes metrics, without new
learned inference, policy decisions, training or MPC. Audit time is separate.

The focused engineering suite passed 119 checks. Independent review found no
blocking issue. These are implementation checks, not evidence of useful control.
Even passing the study earns a strong-history comparison before any connectome
claim; failure returns the work to prediction and planning diagnostics.
