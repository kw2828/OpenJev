# Five spatial value models are ready for a controlled training study

**Implementation and synthetic qualification completed. No empirical training or autonomous result yet.** The new spatial head and four ordinary controls pass independent mathematical checks and a bounded runtime trial. The previous [504-search failure](otto-conditioning-control-results.md) remains unchanged.

The hypothesis is that applying nonlinear processing to neighborhood probability mass before pooling can learn useful action values. This implementation provides the comparison needed to test that hypothesis. It is based on familiar shared-cell transforms, convolution and pooling; it does not establish a novel architecture or a biological, recurrent or world-model advantage.

## What is implemented

All five heads consume the same complete public belief, agent position and supplied sensing length. They preserve zero and subnormalized branches, signed outputs, a common mass-weighted baseline and the original sixteen-branch controller interface. Every fresh model starts at the same function. Immutable NumPy float64 deployment is independent of the Torch training implementation.

| Family | Parameters | Full-batch synthetic update, median ms | Sixteen-branch head inference, median ms |
| --- | ---: | ---: | ---: |
| Spatial neighborhood sums | 737 | 23.37 | 4.19 |
| Same model without neighborhoods | 737 | 17.25 | 2.90 |
| Ordinary CNN | 801 | 72.70 | 9.69 |
| Ordinary dense128 | 1,411,841 | 6.26 | 0.71 |
| Familiar statistics | 225 | 5.49 | 0.67 |

These are warmed synthetic CPU measurements on this machine, with one thread. Updates use float32 and batch 128; inference uses independent NumPy float64 and batch 16. Input validation, feature construction and nested qualification checks/journaling are included. Branch construction, action selection and posterior updates are excluded from head timing. The table is not complete controller latency, a Rust/Python comparison or evidence of task effectiveness. Similar matrix-operation counts do not imply similar elapsed cost.

## Correctness and execution evidence

- Forty independent model tests passed, covering direct neighborhood sums, lossless geometry, boundary behavior, zero/subfloor mass, all parameter counts, local RNG isolation, common initialization, later hidden-layer gradients, biased zero outputs and immutable checkpoint ownership. A separate 1x1-convolution fixture checks the spatial head's known equivalence; it is not another architecture result.
- A CNN fixture specifically detects activations incorrectly leaving the physical board and returning at a later layer. Meaningful nonzero readouts check Torch/NumPy parity, rather than relying only on the common zero-residual initialization.
- Seven harness tests passed, including all-family cost projection, tail batches, exact call allocations, no-overwrite behavior and preserving failures when the clock fails.
- The first engineering attempt's numerical tests passed, but its overall check failed on two test-only lint issues. The original logs remain. After those style fixes, both affected tests and lint passed; the model source was unchanged.
- The frozen runtime trial completed once: **50 disposable synthetic optimizer updates, 95 Torch forwards and 40 NumPy forwards**. All attempted calls returned. Worker time was **2.81 seconds**, enclosing supervised time **3.05 seconds**, with peak process RSS **531.72 MiB**. No native environment calls, external model calls or empirical dataset reads occurred.
- Five saved disposable checkpoints passed float64 backend comparison on sixteen mixed fixtures each. Maximum normalized difference was **5.55e-17**. This limited comparison does not prove action agreement for all future near-tie states or trained checkpoints.

The [frozen plan](../output/otto-spatial-qualification-v1/plan-01.json), [protocol](otto-spatial-qualification-protocol.md), [execution witness](../output/otto-spatial-qualification-v1/execution-witness-01.json), [worker receipt](../output/otto-spatial-qualification-v1/run-01/receipt.json), [timing and parity summary](../output/otto-spatial-qualification-v1/run-01/summary.json) and [original process terminal](../output/otto-spatial-qualification-v1/process-01.terminal.json) retain the exact workload and boundaries. The [first engineering receipt](../output/otto-spatial-qualification-v1/engineering-01/receipt.json) and [style-fix receipt](../output/otto-spatial-qualification-v1/engineering-02/receipt.json) preserve both attempts.

The [independent saved-record audit](../output/otto-spatial-qualification-v1/audit-01/receipt.json) agrees on 2,324 checks: nine source files, seventeen payloads, 166 timed operations, 520 call-journal events, all checkpoint schemas, eighty saved parity pairs and all timing projections. It makes no new forward, optimizer or simulator calls. Actual model execution and measured durations remain evidence from the original pinned worker and supervisor; this audit checks the saved records and arithmetic.

## What this changes next

The proposed five-family, three-seed, 80-epoch study requires **52,800 updates**. Using warmed maximum update times, full and tail batches, final float32 TRAIN/VALID diagnostics and setup/checkpoint costs gives **1,334 seconds**, or **2,668 seconds with a two-times planning allowance**, inside the prospective 7,200-second training budget. This is a feasibility estimate, not a runtime guarantee or admission of the scientific run. It excludes data authentication, shuffling/copies, per-epoch diagnostics, independent float64 saved scoring and full supervision.

Autonomous evaluation is materially more expensive. At the full 2,188-step cap for 72 cases and three seeds per family, the same measured maximum head times alone project to roughly **33 minutes for spatial**, **23 for neighbor-free**, **78 for CNN**, **6 for dense128** and **5 for statistics**. These assume every episode reaches the cap and omit branch construction and state updates. They are workload envelopes, not measured episode lengths or predicted success.

Next is a separately frozen empirical comparison using the same authenticated 5,589 TRAIN and 1,109 exposed VALID rows, identical targets and fresh initialization for every model. The [training integration notes](otto-spatial-training-design.md) identify the lossless data adapter and remaining admission work. Synthetic checkpoints must not initialize it. Scalar error remains diagnostic; any advancement claim still needs fresh autonomous cases, the strong analytic anchor and complete controller costs. The statistics model is especially important: the spatial recipe must earn its additional computation.

The [original design snapshot](otto-spatial-design-review.md) remains unchanged as historical proposal evidence. Implementation: [model](../src/openjev/research/otto_spatial_value.py), [independent tests](../tests/test_otto_spatial_value.py), [qualifier](../scripts/qualify_otto_spatial.py). No scientific criteria, previous outcomes or historical frozen sources were changed.
