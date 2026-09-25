# Provisional observer transfer experiment

Prospective design only, prepared from primary documentation on 2026-09-24. This is not a registration. Held test measurements stay unopened until a subsequent experiment's design is frozen. No dataset archive or measurement array was opened, and no model was run for this review.

## Launch condition

The current [frozen-dynamics observer study](robot-observer-protocol.md) must pass **all five development criteria**, with its original process closed and independent audit agreeing. Otherwise, diagnose the exposed development result before proposing confirmation. A positive average or selected favorable recording cannot substitute for the five-criterion rule.

Even a pass establishes only eligibility to design a fresh experiment. The four robot evaluation recordings are exposed development data and cannot become untouched confirmation again.

## Recommended dataset: CubeSpec Fine Steering Mirror

The [authors' repository](https://github.com/merijnfloren/fsm-benchmark-data) provides a measured physical system with three piezo-actuator voltage inputs and three non-collocated displacement outputs. Hysteresis supplies nonlinear dynamics, although the system is mostly linear. The [dataset license](https://github.com/merijnfloren/fsm-benchmark-data/blob/main/LICENSE) explicitly states **CC BY 4.0**.

The [current official loader](https://github.com/MaartenSchoukens/nonlinear_benchmarks/blob/master/nonlinear_benchmarks/benchmarks.py) specifies approximately 19.5 MB, amplitudes 100/200/300 mV, sampling at 6,400 Hz, and 8,192 samples per period. Each amplitude has six training realizations and three test realizations, with two periods per realization. It declares a 100-sample state-initialization window. The signals contain steady-state periods, not transients.

The future input is the designed applied actuator-voltage waveform. It is neither a future measured mechanical force nor a desired mirror-position trajectory. Forecasting conditional on this waveform does not demonstrate closed-loop control.

Proposed custom split, to register before accessing measurements:

- FIT: training realizations 1-3 at 100 and 200 mV.
- DEV: training realizations 4-6 at those amplitudes. Keep both repeated periods together and preserve each documented orthogonal triplet.
- Leave all 300 mV training records unused. After freezing the complete recipe, evaluate the official tests at all three amplitudes with only their allowed observation prefix.
- The 100/200 mV tests measure new-realization generalization. The 300 mV test is a predeclared **within-system amplitude shift**, not a new environment.
- Moving from the robot to the mirror is **method transfer after retraining**, not transfer of frozen robot weights. Dimensions, sample interval and model order require an explicit new training contract.

This is a custom reduced-training experiment, not a directly comparable official all-amplitude benchmark entry. Randomly splitting periods or overlapping windows would inflate apparent independence. The absence of transients also prevents a claim about recovery from arbitrary initial physical states.

## Strong controls and a bounded question

Ask whether observation-conditioned initialization improves unseen-rollout accuracy over simpler initialization on the **same fitted dynamics**, while retaining a useful complete-request cost. Keep accuracy and the accuracy/latency/storage frontier distinct; charge all prefix work and any fitted normalization.

Include finite-history affine initialization, a learned history encoder, fixed/learned innovation correction, and a recurrent baseline. Include the authors' [28-state linear and nonlinear LFR models](https://github.com/merijnfloren/fsm-benchmark-data#baseline-models) as strong established references; the predominantly linear plant makes these essential. Do not inherit the robot's state dimension or practical margins without a prospective rationale. Freeze budgets, splits, allowed prefix observations and all success criteria before test access.

## Smaller alternative: Cascaded Tanks with Overflow

The [official dataset](https://data.4tu.nl/articles/dataset/Cascaded_Tanks_Benchmark_Combining_Soft_and_Hard_Nonlinearities/12960104/1) is approximately 7.5 MB including supporting media, licensed **CC BY-SA 4.0**. Pump voltage drives a two-tank system; only lower-tank level is the benchmark output. This gives a genuine unobserved state and overflow nonlinearity.

The [primary problem description, section 5](https://pure.tue.nl/ws/portalfiles/portal/268860148/Ifac_Benchmarks_v3.pdf) specifies 1,024 estimation and 1,024 test samples at four-second intervals. The current loader allows a 50-sample initialization prefix. Crucially, estimation and test share the same unknown initial state. This is a cheap secondary identification check, but a weak primary test of initialization transfer or environment shift. An internal DEV split must come only from the short estimation record and be fixed prospectively.

## Relevant established methods

- [Beintema, Toth and Schoukens, 2021](https://proceedings.mlr.press/v144/beintema21a.html): an encoder maps historical inputs and outputs to the initial latent state of a simulated section. This is direct prior art and a strong comparison, not a new mechanism to relabel.
- [Forgione and Piga, 2021](https://arxiv.org/abs/2006.02915): neural state-space identification jointly estimates hidden training states and dynamics with truncated simulation criteria. Any test-state optimization would violate the proposed causal-prefix contract.
- [Revach et al., KalmanNet, 2022](https://arxiv.org/abs/2107.10043): recurrent learned filtering uses a partially known state-space model. Its supervision and model-information assumptions must be matched explicitly; it is not automatically a directly usable baseline for unobserved physical states.

The linked repositories and loader were read as current sources; an immutable commit was not verified. Before a registration, pin their exact revisions, archive/license metadata and permitted file roster without inspecting held-test values. No claim of architectural novelty, Bayesian calibration, stability or robot control follows from this proposed experiment.
