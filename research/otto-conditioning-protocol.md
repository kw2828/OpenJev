# Fixed spatial-input conditioning control

**Prospective, unexecuted protocol.** Qualification and full fitting require their own frozen source, input, runtime and supervisor plans. This first stage tests scalar fitting only. A separately frozen fresh autonomous comparison is required after technical completion and independent audit, regardless of the scalar screen's result. This document admits no model or simulator execution by itself.

## Question and scope

Does a fixed spatial-input gain improve optimization of the same width-eight value network when its initial function, data, targets and update schedule are matched? Compare only gains **1 and 53**, without a gain search. This is an ordinary optimization control, not a new architecture, a coverage intervention or evidence of recurrent/connectome superiority.

The [capacity study](otto-capacity-results.md) passed 6/12 conditions: every TRAIN-excess condition passed and every required VALID improvement failed. The [Bellman study](otto-bellman-control-results.md) passed 19/42 conditions and failed its overall rule. The [coverage collection](otto-coverage-collection-results.md) stopped at its fixed quota without training a mixture. None of these decisions changes. Coverage remains unanswered; this smaller control precedes further collection.

Input conditioning is established optimization practice, not the proposed contribution. For a uniform probability distribution over the 53x53 possible source cells, the spatial vector has Euclidean norm `1/53`; multiplication by 53 gives unit norm. Centering into a padded 105x105 array does not add possible source cells. This motivates the fixed geometric gain without estimating a scale from VALID. It does not establish that actual diffuse states dominate the fitting problem. [LeCun et al., Efficient Backprop](https://leon.bottou.org/publications/pdf/tricks-1998.pdf).

## Fixed inputs and targets

Authenticate the completed original scalar study, its successful parent, independent audit and input/source closure before decoding its cached TRAIN/VALID arrays and corresponding metadata. Bind exact input hashes. Use all **5,589 TRAIN prefixes from 192 teacher episodes** and all **1,109 VALID prefixes from 48 separate teacher episodes**. No old EVAL, DAgger or failed-coverage arrays enter this study, and there is no new collection. VALID is exposed research validation.

The labels remain cached float32 Monte Carlo teacher returns `y=(T-t)/64`, with uniform row MSE. They are realized policy returns, not optimal values or action labels. Repeated prefixes and correlated rows remain present. Set `c0=float32(mean(TRAIN y, dtype=float64))` and require the authenticated existing value; never use VALID to estimate it.

Raw input `x` is the unchanged cached float32 vector of length 11,028: 11,025 centered belief entries followed by `mass*x_position/52`, `mass*y_position/52`, and `mass*lambda/5`. For gain `g`, transform only the first 11,025 coordinates: `S_g(x)=(g*x_spatial, x_context)`. The model is

`f_g(x) = c0*sum(x_spatial) + w_out*ReLU(W_g*S_g(x)+b_hidden) + b_out`.

The baseline uses **raw unscaled mass**, and all three context coordinates remain unchanged. Preserve zero and subnormalized inputs, signed outputs and the existing scale of 64 for physical-unit errors. No normalization, output clipping, feature repair or target rescaling is added. Training applies the gain internally in float32. Final scalar scoring upcasts the original cached float32 inputs and exported weights to float64, then applies the gain internally in float64. Record this arithmetic distinction; do not silently pre-scale and re-round the cached scoring inputs.

## Paired initialization and six fresh fits

Each arm has one ReLU hidden layer of width eight and **88,241 trainable parameters**. Use seeds **10101, 10102, 10103**, seed outer and gain order `[1,53]`. Reproduce the capacity study's fresh width-eight draws: one local CPU Torch generator, hidden normal weights scaled by `sqrt(2/11028)`, final normal weights by `sqrt(1/8)`, and zero biases. From the same base draw at each seed, divide only the first 11,025 columns of the gain-53 first-layer matrix by 53 in float32. Other parameters and `c0` are identical. These are fresh fits, not continuation of capacity or older checkpoints.

Before any full-study optimizer update, initialize and export all six heads and qualify all three initial pairs. Compare their immutable float64 initial readouts on every selected TRAIN row, an all-zero input, and the first TRAIN input multiplied by 0.5 after upcasting. Require every element to satisfy `abs(f_53-f_1) <= 1e-6 + 1e-6*abs(f_1)`. Save initial checkpoints, row identities and all pairing comparisons. Each full pair has 5,591 witness rows, batched at 256: 132 initial readout calls across all six heads and six initial exports. Qualification has 258 witness rows, four initial readout calls and two initial exports. Any pairing failure stops before optimization in that mode. This is approximate functional pairing, not byte identity; division and multiplication rounding remain visible.

Use fresh Adam with learning rate **0.001**, betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay, global gradient norm cap **5**, deterministic CPU float32 and one numerical thread. Fit exactly **80 epochs**, batch **128**, including the short final batch. Independent `numpy.random.default_rng(seed+20000)` instances give both gains the same epoch permutations. There are **44 updates per epoch, 3,520 per fit and 21,120 total**. Save initial/final weights, each epoch-order hash, update records and final predictions. Use the fixed final checkpoint; no early stopping, tuning, seed replacement or validation selection.

Adam's approximate invariance to gradient scaling does not imply equivalent function-space updates after inverse reparameterization. Fixed parameter-space steps, epsilon, global clipping and rounding can alter optimization and implicit regularization. A result therefore concerns this combined conditioning recipe, not an isolated proof of improved Hessian conditioning. [Kingma and Ba, Adam, Algorithm 1 and Section 2](https://arxiv.org/pdf/1412.6980v9).

## Qualification and numerical checks

First run a separate disposable TRAIN-only qualification: `numpy.linspace(0,5588,256,dtype=int)`, seed **10101**, both gains, **six epochs**, batch **128**. This gives 12 updates per fit and **24 overall**. Do not decode VALID, retain qualification weights for scientific fitting, or select a gain using qualification losses.

Qualification applies the initial-pair check to all 256 selected rows plus zero/half fixtures. Final exported NumPy float64 readouts must match a Torch double copy at `atol=rtol=1e-10` on the first eight selected TRAIN inputs plus those two fixtures. The full study repeats final scalar parity on the first eight TRAIN and first eight VALID inputs plus zero/half: 18 rows per checkpoint. Preserve all witnesses. These are scalar checks, not branch-cost or action parity.

Qualification caps are **120 suspend-inclusive seconds, 4 GiB RSS, 128 MiB output**. Its workload projection is `3 * sum(two qualification fit seconds) * (3520/12)`, prospectively required to be at most **400 seconds**, reserving 200 seconds within the full cap for preparation, pairing, scoring, audit-ready output and other worker overhead. Both modes use the same batch size; scaling also repeats fixed fit overhead. This is a cost estimate, not a guaranteed bound. Preserve a failed qualification without changing its limit or projection.

Full-worker caps are **600 suspend-inclusive seconds, 8 GiB RSS, 512 MiB output** and 21,120 optimizer updates. The separate saved-output audit is capped at **600 seconds, 4 GiB RSS, 128 MiB output**. All stages use the pinned existing runtime, with no installation, remote inference, simulator resets, observation branches or policy actions.

## Fixed six-condition scalar screen

Compute the common empirical TRAIN alias floor from byte-identical raw cached float32 feature groups: `F=sum_g sum_(i in g)(y_i-mean_g(y))^2/5589`, using float64 reductions and byte equality to verify hash groups. It is a finite-cohort lower bound, not population aleatoric uncertainty or an optimal-control bound. Deterministic input scaling cannot distinguish raw-identical rows, so this remains a valid common lower bound. Do not claim float32 scaling is exactly invertible or necessarily preserves every distinct-input group.

For each seed, require both unrounded float64 inequalities:

1. `gain53 TRAIN MSE - F <= 0.80 * (gain1 TRAIN MSE - F)`.
2. `gain53 VALID MSE <= 0.90 * gain1 VALID MSE`.

Report all **six** decisions, signed TRAIN excess without clipping or epsilon, all final TRAIN/VALID MSEs, physical MAE, negative-prediction counts, paired changes and all seeds. If the gain-1 excess is zero, the prescribed gain-53 threshold remains zero. The thresholds are practical prospective screens, not significance tests. The scalar flag requires 6/6; it is not the admission rule for autonomous evaluation.

## Evidence, costs and required second stage

Freeze the complete source/runtime/input/qualification closure before a single full launch. Preserve attempted/returned initialization, update, export and scalar-readout operations, unresolved calls and failed outputs. Separate complete worker wall time, preparation, initial pairing, paid fitting, final parity and saved scoring. Include scaling and validation overhead in the appropriate paid operation; equal update counts alone do not establish equal computation.

After successful worker and parent completion, independently reconstruct input identities, initialization relations, initial pairing, every final saved prediction, the common floor and all six conditions. Report actual audit replay work. Optimizer execution and timing remain authenticated execution witnesses.

After technical completion and independent agreement, the intended next stage is a **separately frozen autonomous comparison regardless of scalar pass/fail**. It must retain both gains, all three seeds and the analytic in-bounds controller on matched fresh cases in lambda3/4/5, with complete controller/setup costs, preserved final censored updates and an absolute competence anchor. Fresh seed ranges, cohort size, budgets, deployed branch/action parity and control criteria must be frozen before that stage. No favorable scalar result substitutes for it, and no unfavorable scalar result silently removes a gain. Until it runs and is audited, there is no end-to-end efficacy claim or architecture admission.
