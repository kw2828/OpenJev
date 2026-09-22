# Spatial value models: engineering and runtime qualification

This is a synthetic engineering check of the [reviewed design](otto-spatial-design-review.md). It does not read TRAIN, VALID, evaluation episodes, outcomes or previous checkpoints. No new scientific fit, policy search or biological architecture claim is admitted by this protocol.

Implement all five families exactly: spatial sums (737 parameters), neighbor-free (737), ordinary CNN (801), ordinary dense128 (1,411,841), and familiar statistics (225). Use the exact same public centered belief, explicit physical position and known sensing length. A separate independent test suite checks geometry, boundary sums, initialization, gradients, checkpoint semantics, NumPy/Torch parity and action ties.

## Fixed disposable workload

One CPU thread, float32 synthetic optimization, float64 deployment, local seed 41001 for fixtures and 42001 for model initialization. The disposable baseline is `float32(0.4)`, not the scientific TRAIN-derived baseline. Fixed fixtures mix zero, subnormalized, point, two-point and diffuse mass at board corners, edges and interior positions. Sensing lengths cycle through 3, 4 and 5. A deterministic synthetic regression target exercises gradients; its loss is not an efficacy result. All families receive identical fixtures and targets.

For each family in the declared order:

1. Initialize one head and one Adam optimizer with learning rate 0.001. Record both costs, all parameter shapes and counts.
2. For each batch size 128 and 85, execute two warmup updates and three measured updates. Each update includes input validation, feature construction, forward, MSE, backward, gradient clipping at 5, and Adam. Record every update, including warmup.
3. At each batch size, run one first-call and three warmed no-gradient float32 diagnostic forwards. Record all costs.
4. Export the disposable updated checkpoint once, serialize to NPZ, restore it with pickle disabled, construct an immutable independent NumPy deployment head, and restore a separate Torch head in float64. These setup and file costs are recorded separately.
5. Run one Torch64 and one NumPy64 parity forward on the first 16 fixtures. Require finite values and maximum absolute normalized difference at most `1e-8`. Save both arrays. This is a limited fixture check, not a near-tie guarantee.
6. Run two warmup and five measured NumPy64 deployment forwards on those 16 fixtures. They include validation, unpacking, box sums or convolutions, pooling and readout. This timing excludes observation-branch construction, action selection and posterior updates, so it is not total controller latency.

Exact totals are 10 model constructions (5 training, 5 parity references), 5 optimizer constructions, 50 synthetic optimizer updates/backward calls, 95 Torch forwards (50 update + 40 diagnostic + 5 parity), 5 exports, 5 independent deployment constructions and 40 NumPy forwards (5 parity + 35 timed). Native steps/resets, external model calls and empirical data rows read are all zero. Failed attempts and partial counters remain in the receipt; never replace them with a successful retry under the same plan.

## Bounds and evidence

The frozen JSON plan binds the implementation, independent tests, this protocol, reviewed design, runtime versions, supervisor and suspend-aware clock. The child authenticates all source files and the exact launch. Output is exclusive. Limit the process to 600 suspend-inclusive seconds through the existing external supervisor, with 4 GiB peak RSS and 128 MiB output checked before and after each recorded operation. RSS is a sampled/post-operation limit, not a hard address-space reservation. The process writes attempted/returned call counters and a flushed call journal, preserving partial attempts after interruption. Timing includes these checks and nested call-journal writes conservatively. A terminal supervisor receipt must show exit zero, no timeout and no remaining process group.

Save every disposable updated checkpoint, parity input/output and operation timing, plus file hashes, runtime and a completion receipt. These are engineering artifacts and must never seed scientific fitting. Source files are authenticated again before success. Tests are distinct from this measured run; their synthetic updates are not silently counted as study work.

## Resource projection and continuation

The proposed scientific schedule remains 15 fresh fits: five families, three fitting seeds, 80 epochs on 5,589 rows with batches of 128 (43 full and one 85-row tail per epoch). Diagnostics over 5,589 TRAIN and 1,109 VALID rows use 43+1 and 8+1 corresponding batches. For each family, report measured-update and diagnostic minima/medians/maxima and retain all individual samples.

Project one fit as initialization plus optimizer setup plus `80*(43*max(full_update)+max(tail_update))`, plus one final TRAIN/VALID diagnostic pass and checkpoint costs. Maxima here use warmed samples; first/warmup calls remain in the artifact and timing tables. Multiply the sum across five families by three seeds. A two-times allowance is reported against a prospective 7,200-second training budget. It is a planning estimate, not a statistical upper bound or permission to run: shuffle/copy costs, variable beliefs, per-epoch diagnostics, data authentication and complete run supervision can change it. The projection uses final float32 diagnostics; independent float64 saved-prediction replay would add work. Any later training schedule and overhead accounting require a separately frozen plan before outcome reads.

Also report the measured maximum deployment time times the absolute 2,188-step episode cap, and times 72 cases times three seeds per family. This deliberately exposes the worst-case head-only autonomous workload. It omits branch/update/setup costs and is not an empirical estimate of episode length or success. No full policy evaluation is admitted by this engineering qualification.

Continue to scientific protocol design only if independent correctness tests and checkpoint parity pass and the proposed training cost is feasible. If costs are high, optimize equivalent arithmetic and requalify under a new version before scientific outcomes; do not drop ordinary controls, lower competence criteria or treat synthetic loss as progress in task performance. Recurrent or connectome comparisons remain a later question requiring a demonstrated problem and matched controls.
