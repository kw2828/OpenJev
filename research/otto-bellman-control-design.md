# Proposed observation-backup learning control

Status: design only; no scientific run has started. Runtime qualification and a complete frozen protocol are the next engineering steps. This compares learning objectives in the existing ordinary MLP. The completed return-value study's **0/54** result and all earlier failures remain unchanged.

## Question and fixed inputs

Can branch-based bootstrapping improve autonomous decisions relative to continuing Monte Carlo regression from the same starting model? This follows the model-based value-learning approach in [Loisy and Heinonen, Algorithm 1, Section 3.1 and Appendix C.1](https://auroreloisy.github.io/papers/Loisy2023a_EurPhysJE_drl-benchmark.pdf). It is a positive control, not a claim of novelty or optimality.

Use the three authenticated `mlp8` epoch-80 checkpoints, seeds 10101, 10102 and 10103, from `otto-return-value-v1`. Each starts two independent continuation arms. Include a third, unchanged-checkpoint reference for each seed. Reuse only the 192 original TRAIN teacher episodes and all 5,589 reconstructed current-state prefixes. No exploratory, DAgger, VALID or EVAL states enter fitting or target generation. Existing VALID may be reported descriptively under a declared scope, never used for selection.

Keep the 11,028 raw centered-belief and mass-scaled public-context features, width-eight biased ReLU MLP, stored float32 `c0`, known kernels, sixteen explicit branches and strict first-eligible near-tie rule unchanged. Deployment remains float64 arithmetic on upcast float32 weights.

## Candidate recipe

Continue for 40 fixed epochs using float32 training, fresh Adam optimizers at learning rate 0.001, batch 128 and gradient norm cap 5. Old optimizer states were not saved, so neither arm claims optimizer continuation. Both retain identical weights and `c0` at initialization. Pair permutations using `numpy.random.default_rng(seed+30000)`. Use uniform row MSE and the fixed last checkpoint; no learning-rate search, early stopping or checkpoint selection.

- **Continued MC:** retain `y=(T-t)/64` from each complete teacher episode.
- **Observation backup:** initialize a frozen target network from the same checkpoint. Before epochs 1, 6, 11, ..., 36, materialize one target per TRAIN state using that network; reuse it for five epochs. After each five-epoch block, copy the current learner into the target network for the next block. Target generation has no gradients.

For physical target value `C_bar=64*f_bar`, use

`y_backup(b) = [1 + min_eligible_a sum_h w[a,h]*C_bar(z[a,h])] / 64`.

Evaluate the target network in the qualified float64 deployed arithmetic. Save float64 targets and their float32 training casts, all refresh-checkpoint identities and target statistics. Stream bounded batches rather than retaining all branch features. Targets are **self-generated within the backup arm**; they are not shared labels or ground-truth action values. This compares complete learning procedures, not architectures under identical targets.

## Numerical and resource limits

Preserve `w=max(raw_mass,1e-10)` and `z=u/w`, including kernel-origin zeros. Evaluate the biased MLP at zero/subnormalized branches without overriding its output. Do not renormalize weights, clamp signed values, add discounting, or replace tiny branches. Found continuation contributes zero through the existing terminal branch semantics; this does not impose `f(0)=0` or a learned point-mass terminal value. Native found/censored final observations remain assimilated.

This is an undiscounted approximate backup. A delayed target network does not establish contraction or convergence. Stop and preserve nonfinite failures; retain finite target growth and negative-value diagnostics without adaptive repairs or restarts. Qualification must fix runtime, memory, output and call caps before execution.

Equal epochs and updates **do not equal total compute**: the backup arm additionally constructs branches and evaluates target networks. Charge every refresh and report the equal-update comparison as such. Any compute-matched comparison needs a separately frozen MC update schedule, derived from TRAIN-only timing qualification; do not waste compute, hide target costs or choose budgets from evaluation results.

## Evaluation and interpretation

Freeze fresh paired evaluation seeds before continuation; never reuse the exposed old cases. Compare both continuations, all three unchanged checkpoints and analytic control under identical public inputs, complete inference costs and retained failures. Require autonomous competence before advancing. Numerical parity and improved training loss alone do not establish policy improvement, a memory benefit or optimal planning.
