# Fixed-data scalar regression capacity screen

Status: prospective. Qualification and the full study require separately frozen plans, source hashes, runtime identities and exclusive outputs. This document records the intended comparison; no capacity result or autonomous-control admission exists yet.

## Question and scope

Can larger ordinary networks fit the same teacher-return data more accurately than a newly initialized width-eight network? This isolates a capacity-and-optimization recipe on common scalar targets. It does not test search actions, new state coverage, recurrence or biological wiring. The prior scalar study's 0/54 and Bellman-control study's 19/42 failed decisions remain unchanged.

The closest OTTO benchmark used three hidden layers of 1,024 units for its larger isotropic case, with 11,025 belief inputs and 13,390,849 parameters. Its learning algorithm and data differ from this screen. That is a reason to test ordinary capacity before interpreting our small-network failures, not evidence that width caused those failures or that this experiment reproduces the benchmark. [Loisy and Heinonen, Section 3 and Appendix C, Table 5](https://auroreloisy.github.io/papers/Loisy2023a_EurPhysJE_drl-benchmark.pdf).

## Fixed data and target

Authenticate the completed `otto-return-value-v1` plan, worker, supervisor and independent audit before decoding its saved `train-data.npz`, `valid-data.npz` and corresponding row metadata. Bind their exact hashes in the new plan. Preserve all 5,589 TRAIN prefixes from 192 naturally found teacher episodes and all 1,109 VALID prefixes from 48 separate naturally found teacher episodes. Do not decode old EVAL or DAgger arrays or trajectories. No new collection is part of this study.

The common target is the saved float32 Monte Carlo teacher return `y=(T-t)/64`. All rows have equal loss weight. Prefixes from the same episode are correlated. These are realized returns of the analytic teacher, not optimal values or branch labels. VALID is already exposed research validation, not an untouched generalization test.

Use the exact cached float32 feature vectors, with 11,028 entries: the flattened centered 105x105 raw belief followed by `mass*x/52`, `mass*y/52` and `mass*lambda/5`. No normalization, standardization, clipping, augmentation, feature repair or higher-precision feature reconstruction changes the fitting or primary scoring inputs. Preserve the existing scalar `c0 = float32(mean(TRAIN y, dtype=float64))` exactly and verify it against the authenticated cache. Never estimate it from VALID.

## Empirical feature-alias floor

Group TRAIN rows by identical canonical C-order float32 feature bytes. Hashes may index groups, but equality must be confirmed from the bytes. Retain all rows, including repeated states. Let `G` be these groups and `N=5589`. Compute in float64:

`F = sum_g sum_{i in g} (y_i - mean_{j in g}(y_j))^2 / N`.

This is the empirical lower bound for a deterministic regressor of those exact inputs on this observed cohort. It is not a population aleatoric-noise estimate, an optimal-control bound or evidence about unobserved states. Report total rows, unique and duplicate-group counts, rows in duplicate groups, groups with conflicting targets and the resulting floor. The authenticated cache preserves every group membership, target and contribution for independent reconstruction. Do not merge TRAIN and VALID groups. Compute final predictions from the same cached float32 inputs upcast to float64, so the bound and model use identical information.

Report signed TRAIN excess `MSE-F` without epsilon or clipping. Any negative floating-point residual is reported explicitly. A zero narrow-model excess does not trigger an alternative criterion: the prescribed wide-model excess must then be at most zero.

## Nine fresh fits

Every network predicts normalized value `c0*mass + residual(features)` with ReLU hidden layers, biased linear scalar output and no output intervention.

| Family | Hidden layers | Trainable parameters |
| --- | --- | ---: |
| mlp8 | 8 | 88,241 |
| mlp128 | 128 | 1,411,841 |
| deep128 | 128, 128, 128 | 1,444,865 |

Use seeds 10101, 10102 and 10103 independently for each family. A local CPU Torch generator draws each hidden weight from a standard normal scaled by `sqrt(2/fan_in)` and the final weight by `sqrt(1/fan_in)`; every bias starts at zero. Draw the first matrix in eight-row blocks, preserving identical first-eight rows across all families and identical first-128 rows across both wide families at a given seed. Record initial hashes. Shapes and later draws differ, so a paired seed does not mean identical initial functions. This initialization differs from the earlier 0.01-scale recipe: the fresh narrow control is essential, and direct comparisons to old fitted checkpoints are not this study's capacity test.

Fit for exactly 80 epochs with fresh Adam, learning rate 0.001, batch size 128, gradient norm cap 5, float32 arithmetic, deterministic operations and one CPU thread. Use ordinary uniform-row MSE. Each family at a given seed receives the same epoch permutations from an independent `numpy.random.default_rng(seed+20000)` stream. Preserve the final short batch. There are 44 updates per epoch, 3,520 per fit and 31,680 overall. Equal updates are not equal computation across these networks.

No early stopping, learning-rate search, seed replacement or checkpoint selection is allowed. Save the final checkpoint, every epoch-order hash, actual update counters, fixed execution/loss records and final TRAIN/VALID predictions. Report all nine final MSEs, physical-unit errors, negative-prediction incidence, parameter counts and measured fit/scoring costs. Intermediate online batch losses are not substituted for final checkpoint MSE.

## Numerical and runtime qualification

Before the full study, run a separate disposable TRAIN-only qualification: indices `numpy.linspace(0,5588,256,dtype=int)`, all three families at seed 10101, six epochs, batch size 128, two updates per epoch. Its 36 total updates exercise initialization, Adam, export, scalar scoring and accounting. Each final qualification model predicts the 256 selected rows in one batch, for three saved-prediction calls overall. It must not decode VALID, continue its weights into the study, or select a family using fit quality.

Qualification compares exported float32 weights upcast to NumPy float64 against a Torch double copy on the first eight selected TRAIN cached input states, followed by an all-zero input and the first selected input multiplied by 0.5: ten rows per model. The full study repeats scalar parity for every final checkpoint on the first eight TRAIN and first eight VALID cached rows followed by those two fixtures: eighteen rows per model. Each feature vector is upcast to float64 before fixture scaling. Use `atol=rtol=1e-10`; retain every comparison and stop on failure without changing tolerances. These fixed scalar witnesses do not establish action or branch parity. There are no branch, policy, action-selection or native-simulator calls.

The qualification must finish within 120 suspend-inclusive seconds, 4 GiB RSS and 128 MiB output. Its predeclared full-fit projection is `3 * sum(three qualification fit seconds) * (3520/12)` and must be at most 1,200 seconds, reserving 600 seconds of the full limit for setup, final scoring, verification and output. Qualification and full fitting now use the same batch size. Scaling paid fit time also scales its fixed initialization/export overhead; this is a workload estimate, not a guaranteed upper bound under different cache, memory or runtime conditions. Report all three measured fit times and the projection. The actual full-study cap remains decisive. Preserve any failed qualification and do not silently alter the projection or runtime to pass it.

## Fixed twelve-condition screen

For each wide family and each paired seed, require both:

1. Final VALID MSE is at most `0.90 * mlp8 VALID MSE`.
2. Final TRAIN excess is at most `0.80 * mlp8 TRAIN excess`.

These are exactly 12 direct float64 inequalities with no rounding allowance. They are prospectively chosen practical screens, not significance tests or power-calibrated thresholds. Show the two metrics, floor, denominators and each decision for every paired comparison. Preserve both wide families and every failed cell; do not select the better family after seeing results. All 12 conditions must pass for this screen's continuation flag.

A pass motivates a separately frozen fresh autonomous comparison; it does not establish competence, better action ranking, a useful teacher or architectural novelty. A failure rejects this fixed capacity recipe under this data and budget, not all larger networks. No branch or autonomous test is automatically added. Better scalar regression can coexist with worse action selection.

## Execution and independent verification

Freeze protocol, implementation, tests, runtime/package manifest, input lineage and qualification receipt before the single full-study launch. Preserve original sources and failed attempts. Use the existing CPU runtime without installation or external calls. The full worker is capped at 1,800 suspend-inclusive seconds, 8 GiB RSS, 1 GiB output and 31,680 optimizer updates; qualification has its own output and counters. Journal attempted and returned atomic updates, exports and scalar readouts, preserving unresolved work on failure.

After successful worker and supervisor completion, independently verify cache and row identities, all final saved predictions from the cached inputs and exported weights, the alias floor, scalar metric reductions and all 12 conditions. Report independent replay computation honestly. Actual optimizer execution and timing remain source-bound execution evidence. Incomplete work or audit disagreement prevents a completed screen or continuation claim.
