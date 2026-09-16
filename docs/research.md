# Research directions toward ICLR 2027

Status: research infrastructure and hypotheses, not a novelty or submission-readiness claim. OpenJev's current deployed model is still a pretrained transformer used as a candidate scorer. None of the experimental architectures below is trained or connected to the public demo.

The [initial Jev source review and baseline notes](jev-background.md) are preserved separately.

The [Bayesian calibrated-decision study](bayesian-rlcd.md) adds a fitted outcome model and real Doom experiments, including a preserved measurement failure and a separate corrected protocol. It distinguishes TypeSafe RLCD, contrastive-distillation RLCD, and public RLCR.

## A focused research question

**Can uncertainty determine how much computation and memory a bounded decision needs, while improving the quality/latency tradeoff under distribution shift?**

The proposed system first scores candidates cheaply, then may spend a fixed remaining budget on recurrent latent updates or short world-model rollouts. Measure the resulting decisions, coverage, and compute cost. Compare against fixed-depth computation and simple confidence/entropy gates. A combination of existing components is not automatically novel.

A potentially useful contribution is a controlled study of when adaptive computation helps or fails as candidate sets, instructions, and observation quality change. A stronger contribution would need a new selection/calibration method with justified assumptions, or a reproducible improvement beyond strong matched-budget baselines. Doom alone and a Rust rewrite alone are unlikely to support the full research claim.

## Available components

| Component | Implemented now | What still needs evidence |
| --- | --- | --- |
| Split conformal prediction | `openjev.research.conformal.SplitConformal`: 1 minus correct-label probability, finite-sample quantile, ties included, empty sets preserved, calibration-unit overlap and signature checks | Real labeled calibration/test data, exchangeability justification, coverage under shift, useful set sizes |
| Semantic grouping | Entropy of probability mass aggregated over explicit equivalence groups | Task-valid equivalence groups and comparison against ordinary action entropy; this is not sampled-text semantic entropy |
| Connectome-inspired controller | `SparseCircuit`: trainable sensory/inter/command/motor-style wiring with explicit recurrent state | Training, dense and shuffled-topology controls, robustness and latency evaluation; no biological connectome is imported |
| Latent transformer | `LatentTransformer`: one shared attention block, selectable depth 1-16, optional temporal state, candidate readout | Training across depths, quality/compute curves, stopping rule evaluation; not pretrained Huginn |
| Recurrent world model | `RecurrentWorldModel`: GRU observation update, action-conditioned latent transition, feature/reward/termination heads, candidate readout | Learned dynamics, rollout error, planning integration, actor/critic or other RL training; not Dreamer or an RSSM reproduction |
| Rust/Python comparison | Equivalent float64 candidate softmax, full-vocabulary candidate mass, and entropy; output parity checked before reporting speed | HTTP/serialization and full inference benchmarks with identical weights, precision, prompts and hardware |

The architecture modules require the `train` extra. They accept numeric observation features and candidate vectors. A trained text encoder/adapter is still required to connect arbitrary English questions and answers to them. Hidden states are passed explicitly; start each episode with `None`, detach between truncated training sequences, and never share state between users or episodes. The sparse module uses dense masked operations, so fewer active edges do not imply faster execution.

```python
import torch
from openjev.research.architectures import LatentTransformer

model = LatentTransformer(observation_dim=11, candidate_dim=8)
observation = torch.randn(2, 11)
candidates = torch.randn(2, 6, 8)
logits, state = model(observation, candidates, depth=2)
# Random weights: this exercises the interface, not a useful policy.
```

## Conformal prediction: the claim we can support

For exchangeable held-out examples and a frozen scoring rule, ordinary split conformal prediction targets **marginal label coverage**, not the safety of an action, correctness conditional on acting, or survival of an entire episode. Candidate probabilities need not already be calibrated. The correct label must exist in the candidate set; omitted answers cannot be recovered by calibration.

Calibration uses scores `1 - p(correct_id)` and rank `ceil((n+1)*(1-alpha))`. When that rank exceeds `n`, all candidates are included. A singleton can be routed to an action by an application, but overall coverage does not establish the accuracy of that selected subset. Report selective error separately. Multi-answer and empty sets are real outputs, not silently replaced by argmax.

In Doom, adjacent frames are dependent. Split by whole episodes and preselect one timestep per episode using a frozen rule for an initial marginal-decision study. For a trajectory guarantee, define an episode-level score and derive the corresponding set construction separately. Held-out episode seeds alone do not repair policy-induced distribution shift: an adaptive controller changes the states it visits. Never reuse calibration units as a final test set, select thresholds after seeing confirmation results, or infer guarantees from the eight development examples.

The synthetic sanity check uses 500 calibration and 2,000 test examples drawn IID from a known six-class distribution. It checks mechanics only; see `evidence/conformal-synthetic.json`. It says nothing about Qwen or Doom quality.

```sh
uv run --extra train --extra dev pytest -q tests/test_research.py
uv run python research/conformal_sanity.py
```

## Experiment sequence and stop rules

1. **Freeze the contract.** Record source/checkpoint/prompt hashes, candidate construction, input modalities, train/calibration/development/confirmation splits, task labels, hardware, seed list, and the inference/training budget. The initial numeric modules are smoke-tested only. Keep a development run separate from a later frozen confirmation run.
2. **Establish baselines.** Fixed Qwen scoring; the existing tiny MLP; feed-forward and GRU models with comparable parameter counts; fixed-depth latent computation; argmax, probability-threshold and categorical-entropy gates. Add matched-parameter and matched-wall-time comparisons because they answer different questions.
3. **Separate topology from sparsity.** Compare circuit wiring with dense recurrence, random equal-edge masks, and degree-preserving rewiring under equal training data. Use sensor dropout, delayed observations, variable frame skip, and unseen scenarios. Measure both robustness and actual runtime.
4. **Separate memory from depth.** Cross temporal state on/off with recurrent depth 1/2/4/8. Use identical candidate encoders and readouts. Calibrate each fixed depth separately. If a gate chooses depth adaptively, fit and evaluate the complete selected predictor on separate data; selecting among calibrated predictors does not automatically retain coverage.
5. **Learn, then test dynamics.** Start with one-step feature/reward/termination prediction and multi-step error on held-out episodes. Compare reactive GRU against fixed-horizon and adaptive-horizon planning. Keep observed future states out of imagined rollouts. Stop planning experiments if dynamics error grows faster than the baseline or planning consumes its latency budget without a quality benefit.
6. **Evaluate beyond Doom.** Add English tasks with explicit labels, including unseen candidate descriptions, reordered/renamed IDs, equivalent descriptions, omitted correct answers, and unsupported instructions. Use independently sourced test examples. Keep privileged-state Doom separate from a future pixel-input study.
7. **Bound exploration before confirmation.** Start with at most three architectures, three development training seeds, 50,000 environment steps per seed, and six wall-clock hours total on existing hardware. This is a proposed ceiling, not a launched run. No paid compute is needed for the included checks. Advance at most one configuration to a preregistered confirmation comparison with at least five fresh training seeds and paired evaluation seeds. Retain failures and report confidence intervals resampled at the independent episode/seed level.
8. **Stop on a negative result.** If gains vanish under compute matching, rely on privileged information unavailable to the comparator, or come from tuning on confirmation data, do not present the configuration as an improvement. A well-controlled negative result can still identify a useful limit; it does not guarantee conference acceptance.

Primary metrics: candidate accuracy, set coverage and mean size, selective error versus action rate, episode return/kills/survival, instruction violations before/after hard masks, and p50/p95 end-to-end latency. Also record training time, peak memory, environment steps, cold starts and model-load time separately. Do not conflate game time with wall time or hard-coded pacifism with learned instruction following.

## Rust versus Python: what the benchmark measures

```sh
docker build -t openjev:hosting .
docker build -f research/Dockerfile.benchmark -t openjev:benchmark .
docker run --rm --cpus=2 --memory=1g openjev:benchmark
```

`research/benchmark_scores.py` compares the same float64 inputs: 32 rows, 151,936 vocabulary logits, six candidates, three warmups, and 11 timed batches. It checks every probability, candidate mass and entropy against Rust with an absolute tolerance of `1e-10`. Inputs include tied and extreme logits. Timed sections exclude process startup, input parsing and file I/O. NumPy already executes its numerical kernels in native code. Rust uses scalar standard-library loops at release optimization level 3, not a GPU implementation.

The initial run measured median **0.550 ms per row for NumPy versus 0.204 ms for Rust**, about **2.70x** for this kernel. Maximum absolute output difference was `1.12e-18` or less. The raw receipt is `evidence/rust-python-score-kernel.json`. This is an isolated score-processing microbenchmark, not a benchmark of the Python web service against a Rust service, and not a measurement of model inference speed. The benchmark runs Rust then Python on a shared workstation with other Docker services running; background load, order, vectorization, allocation and thermal effects remain limitations. A publication comparison needs alternating order, multiple machines/runs, and complete request latency with fixed model quality. A hybrid Rust runtime is worth implementing only if profiling identifies meaningful overhead outside the existing native model kernels.

## Related work and novelty boundary

- [Angelopoulos and Bates, conformal prediction tutorial](https://arxiv.org/abs/2107.07511): foundation for the split-conformal baseline, not our methodological contribution.
- [Barber et al., Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415): distribution drift and nonexchangeability require additional treatment; arbitrary online policy adaptation is not covered by vanilla split conformal.
- [Neural Circuit Policies](https://ncps.readthedocs.io/en/latest/quickstart.html) and [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439): established biologically inspired recurrence. Our masked tanh scaffold is neither an LTC implementation nor a biological reconstruction.
- [Geiping et al., recurrent-depth latent reasoning](https://arxiv.org/abs/2502.05171): shared latent-depth computation already exists.
- [DreamerV3](https://arxiv.org/abs/2301.04104): learned world models and imagined-policy optimization are established. Our GRU scaffold does not reproduce its training algorithm.
- [Looped World Models, June 2026](https://arxiv.org/abs/2606.18208): shared transformer recurrence for latent environment prediction is already proposed. Combining recurrence with a world model is not a defensible new claim by itself.

This is an initial primary-source review as of September 16, 2026, not an exhaustive novelty audit. Before choosing a paper claim, audit uncertainty-based adaptive computation and conformal decision/planning literature in depth and compare directly with the closest methods.

## Memory ablation decision

The [frozen 1,980-episode history ablation](memory-ablation.md) improved utility over the original head but failed its full continuation gate against the same-dimensional current-only control. Adaptive-compute and second-environment efficacy experiments were not launched. This closes the current bounded experiment; the architectural hypotheses above remain unvalidated.
