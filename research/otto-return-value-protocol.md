# Matched scalar-value learning for public-belief odor search

Status: prospective protocol. Execution is admitted only after the completed source review, engineering checks, exact plan and source closure are committed and pushed. No training or native episode is authorized by this document alone before that freeze.

## Question and prior evidence

Does a small minimum-linear value model improve autonomous search or its complete computation cost relative to two parameter-matched neural value models, when all learn identical analytic-policy returns and use identical observation branches?

The preceding symmetry/action-preference experiment failed 2/54 conditions. Its saved trajectories showed spatial oscillation in 164/213 failed learned searches without exactly-zero posterior mass or exact repeated position/posterior-hash states. That motivates this readout experiment but establishes neither a cause nor a memory deficit. Every previous result and continuation decision remains unchanged.

This pilot estimates the analytic controller's value, not optimal value. A heuristic policy's value need not have the minimum-linear structure of an optimal finite-horizon POMDP value. Full observation backups and alpha-vector approximations are established methods, including [Loisy and Heinonen, Sections 2 and 3](https://arxiv.org/html/2302.00706v2). Passing this pilot would admit stronger testing, not establish novelty, recurrent memory, biological wiring or general robotic performance.

## One common scalar dataset

Use the authenticated teacher TRAIN and VALID episodes from `otto-symmetry-head-v1/run-01`. TRAIN consists of 192 naturally completed teacher episodes, 96 each at sensing lengths three and four, and all 5,589 pre-action states. Reconstruct each float64 public belief from the saved reset and observation/action packets and check every recorded posterior hash. Do not invert the old rounded square-root features. Authenticate the entire prior run, but exclude its DAgger and EVAL trajectories from every target, input, fit and selection decision.

At each pre-action index `t`, use the scalar target `(T - t) / 64`, where `T` is that teacher episode's completed search length. Use every prefix, with uniform row squared error. No 64-prefix selection, episode-length weights, log-return transform or relative action preferences. Duration-dependent sampling or weights could change the conditional return being estimated. Correlated prefixes do not become independent episodes.

VALID is the 48 separately collected teacher episodes, 24 per supported sensing length. Use all their pre-action states for descriptive validation, never training or checkpoint selection. Its row count is derived from the closed episode records. Require complete naturally found TRAIN and VALID episodes before making these uncensored-return targets; any missing or censored input stops preparation without deletion, replacement, fabricated tail value or criteria changes. Future collections would need a separately declared censoring treatment.

Compute `c0 = float32(mean(TRAIN normalized targets, dtype=float64))` once. Store it and use the same value in every model. Targets are integer multiples of 1/64 and are exactly representable in float32. Save the complete row identity, pre-action public packet, posterior hash, target, dataset arrays and hashes. A current state supplies one target, not labels for its sixteen hypothetical branches.

## Matched model definitions

For a centered raw belief `v` of shape 105x105, its mass `m`, public position `(x,y)` and supplied sensing length `lambda`, all models receive the same 11,028 features:

`[v.flatten(), m*x/52, m*y/52, m*lambda/5]`.

No square root, normalization repair, standardization, hidden source, time, seed, action history or future packet enters the model. The mass-scaled context keeps zero/subnormalized inputs meaningful. All models predict normalized value `c0*m + residual(features)`; deployed physical value is 64 times that prediction.

| Family | Residual model | Trainable parameters |
| --- | --- | ---: |
| min8 | Minimum of eight bias-free linear planes | 88,224 |
| mlp8 | Width-eight ReLU layer and linear scalar output, with biases | 88,241 |
| homogeneous8 | Same width-eight ReLU network without biases | 88,232 |

All predictions remain signed. Do not apply a positivity clamp, output softplus, extra zero-input override or fallback. Negative predictions and minimum-plane usage are diagnostics, never grounds for a reset or model replacement. The common mass-only baseline is target centering, not an architecture-specific analytic or distance initialization.

Fit seeds are 10101, 10102 and 10103, producing nine models. For each seed, a fresh local Torch generator draws the common first matrix `randn(8,11028)*0.01`; min8 uses those planes and both MLPs use that same first matrix. Both MLPs use the next `randn(8)*0.01` draw for output weights. Ordinary-network biases start at zero. Paired initializations and all subsequent training settings are fixed.

## Fitting and deployed arithmetic

Train each model for 80 epochs with Adam, learning rate 0.001, batch size 128, gradient norm cap 5, float32 training, one CPU thread and deterministic Torch operations. Use uniform mean squared error in normalized movement units. Each family at a given seed receives identical epoch permutations from `numpy.random.default_rng(seed + 20000)` and identical targets. No augmentation, replay collection, Bellman bootstrapping, tuning or checkpoint selection is part of this pilot. The only eligible checkpoint is epoch 80. There are exactly 31,680 optimizer updates across nine fits.

Record every epoch-order hash, actual optimizer update, fixed ten-epoch diagnostic and final checkpoint. Store final TRAIN/VALID scalar predictions and negative-value incidence; record minimum-plane usage for min8 without intervening. Intermediate fitting diagnostics remain execution records unless independently reconstructible from saved weights. No intermediate checkpoint is promoted based on validation.

Store learned weights and baseline as float32 arrays. Deploy the qualified explicit branch calculation in float64 with NumPy float64 upcasts of those exact weights. Compare against a Torch double copy of each learned model on the first eight TRAIN and first eight VALID prefixes, including every one of their sixteen branch scalar values, four action costs and final eligible action. Fixed scalar/cost tolerance is `atol=rtol=1e-10`; selected actions must agree exactly. Save these comparisons. A failure stops before autonomous evaluation; do not widen tolerance or switch routes.

Construct all four actions and four nonfound observation branches on demand. Preserve kernel-origin zeros and `weight=max(raw_mass,1e-10)` without renormalizing weights. A biased network's value on a zero branch is still evaluated and weighted. Score `1 + sum(weight * physical_value)` and then choose the first eligible numeric action ID strictly within 1e-10 of the eligible minimum. Score blocked stay-and-observe actions before restricting selection to in-bounds movement. Do not deploy the fused homogeneous shortcut in this pilot.

## Fresh autonomous comparison

Use the same 53x53 sampled-source environment, four hit categories, R_dt=2 and Euclidean sensing model. Sensing lengths three and four are training-supported. Length five is an unseen supplied kernel on the same grid, not an unknown observation model.

Three template resets at seeds 11500001 through 11500003 must exactly match the prior saved kernels and initial-hit mixtures. Eight length-five mechanical cases at seeds 11400001 through 11400008 take at most 32 prescribed steps, cycling actions `(0,2,1,3)` until found. These check the unchanged public/native filtering and are not efficacy cases.

EVAL starts at seeds 11100001, 11200001 and 11300001 for sensing lengths three, four and five respectively. Each setting uses 24 cases, `initial_hit=1+case%3` and `block=case//3`. Compare all nine learned models and `analytic_inbounds` on each case: 720 episodes total. Rotate arm order by the global case index modulo ten. Pair sampled sources and random uniforms by channel/index, while retaining the different observations produced by different paths. These cases are disjoint from all previous odor-search seed ranges.

Every episode ends on finding the source or at 2,188 moves. Censored searches contribute the full horizon; assimilate and record the final observation in both cases. Save complete public trajectories, posterior hashes, sixteen branch masses and values, four raw action costs, eligibility, choices, native events and evaluator-only source/draw evidence. The model receives no evaluator-only data. Record oscillation and posterior-mass diagnostics descriptively without policy intervention.

## Complete cost and fixed continuation conditions

Controller time includes public-filter initialization, branch construction and copies, centering/context features, scalar readout, reduction/selection, every posterior update, and actual inference setup. Retain and charge any unused analytic distance table inherited by a learned actor. Each head's load is allocated over its 72 evaluation episodes; shared model-module setup is allocated over 648 learned episodes. Native simulator time, fitting and data preparation are reported separately. Episode `environment_seconds` measures native steps; reset and template setup costs remain separate entries in the complete native-operation ledger. Exclude only measured nested artifact I/O from controller timing, without double counting. One rotated CPU run does not establish repeated deployment latency.

Use the saved native initial-hit mixture after averaging equally within each hit stratum. Family means average all three fitting seeds. Report all ten arms, three settings and eight paired blocks, including every failure. These 54 prospective conditions apply to min8 and all are required:

- Competence, 18 conditions: each fitting seed in each setting has weighted success at least 0.95 and capped moves no more than 1.05 times analytic control.
- Utility/computation, 12 conditions: in each setting the family has success at least analytic control, moves no more than 1.05 times analytic control, complete controller time at most 0.80 times analytic control, and every individual fit is faster than analytic control.
- Architecture, 24 conditions: in each setting against each neural control, the family has at least its success, at most 0.95 times its capped moves, strictly fewer moves on at least six of eight paired blocks after averaging paired fitting seeds, and no greater complete controller time.

Descriptive gains that miss any required condition do not admit architectural escalation. A simple homogeneous control explaining a gain is reported directly. No previous failed gate is revised.

## Execution, resources and audit

Freeze and push protocol, exact plan, implementation, tests, source review and engineering evidence before data preparation, fitting or a native call. Bind the completed symmetry plan, worker, supervisor and independent audit, their 121-source closure, all original payload hashes, the qualified branch module and the new code. Use the existing Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1 and Torch 2.14.0 runtime and installed-distribution manifest without installation or changes. No TensorFlow, external model API or paid service is allocated.

One exclusive supervised worker is capped at 5,400 suspend-inclusive seconds, 8 GiB RSS, 6 GiB output, 731 native resets, 1,575,616 native steps and 31,680 optimizer updates. The larger output allowance accounts for full sixteen-branch evidence at the worst-case episode horizon. Journal attempts and returns around each native operation, inference readout and atomic optimizer update. A training update encloses its forward pass, loss, backward pass, gradient clipping and optimizer step; these are not independent nested operation counts. Preserve pending calls and all failures. No retry, replacement case, late checkpoint choice or budget extension is allowed after launch.

Only after successful worker and supervisor completion, run the independent saved-output auditor with its own exclusive output, 1,800-second native clock, 4 GiB RSS and 128 MiB output cap. Independently reconstruct public filtering, scalar datasets/targets/baseline, saved checkpoint predictions, branch arithmetic, choices, complete episode/cost aggregation and all continuation conditions. Count local checkpoint replay computation explicitly. Original optimizer trajectories, timing truth and native random execution remain authenticated execution evidence, not independently repeated science. Disagreement prevents an efficacy claim; a stopped attempt remains stopped.
