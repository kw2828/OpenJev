# One candidate after static card recall: action-conditioned noisy-state prediction

Prospective assessment, 2026-09-19T10:20:00.239254+00:00. No environments, models, training, seeds or candidate data were run or allocated. No current calibration outputs were opened for this note. Its fixed gate is unchanged; preliminary results communicated by the parent are not treated here as an independently verified final report.

**There is not yet a justified next large architecture study.** One useful small candidate is an action-conditioned forecasting comparison on native **POPGym NoisyPositionOnlyCartPoleHard**. The question is whether learned persistent state adds useful information beyond a short public history and a conventional filter. This is a test of a possible need, not a proposed novelty claim. Do not reopen the stopped clean Pendulum, Reacher or static-card recipes, or manufacture a hidden dynamics switch to favor a new gate.

## What the primary sources establish

1. **Recurrent Kalman Networks (Becker et al., ICML 2019)** already learn a predict/update filter in a factorized latent state, retaining information such as velocity that is not directly observed. The paper includes action-input-conditioned pneumatic-joint forecasting. This is directly relevant prior art for learned uncertainty-weighted state correction, not evidence that our card KDN implements that model or that its uncertainty is calibrated. [Paper and code links](https://proceedings.mlr.press/v97/becker19a.html), [full paper, methods and pneumatic-joint prediction experiment](https://proceedings.mlr.press/v97/becker19a/becker19a.pdf).
2. **PlaNet (Hafner et al., ICML 2019)** separates history-based inference from action-only latent prediction and uses the latter for planning. Its published experiments also show that useful data collection matters. It motivates testing action-conditioned forecasts before making a control claim; it does not justify assuming an offline forecast improvement will improve native return. [Paper](https://proceedings.mlr.press/v97/hafner19a.html), [full paper, sections 2-3 and experiments](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf).
3. **POPGym (Morad et al., ICLR 2023)** provides small partially observed tasks and multiple memory baselines. Its benchmark comparisons do not establish that a short explicit history or supplied-physics filter fails on the pinned task below. Historical PPO scores also do not directly compare to this proposed supervised forecast screen. [Original paper](https://arxiv.org/html/2303.01859v1), [official repository](https://github.com/proroklab/popgym).

These sources establish that the general mechanism is conventional. The missing evidence is a concrete, reproducible failure of strong simple inference controls on data relevant to decisions.

## Exact candidate and why it differs from cards

At the already-reviewed POPGym commit `410d5aa626dae8024f498354d8781a0d1870c399`, the public observation contains cart position and pole angle; velocities are absent. Actions move the physical state, so remembering an old observation unchanged is generally stale. The noisy Hard class adds independent Gaussian observation noise with standard deviation 0.3 to both coordinates, then clips to the observation-space bounds, including at reset. **This noisy Hard class inherits the base 200-action limit, not the separate noiseless Hard class's 600-action limit.** The implementation, rather than a similarly named benchmark description, controls. [Pinned noisy class](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/envs/noisy_position_only_cartpole.py), [pinned parent](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/envs/position_only_cartpole.py).

Keep the native task, noise, early termination and horizon. Do not introduce a new blackout, gain change or longer episode. Pin the underlying Gymnasium CartPole revision too before implementation. `get_state()`, `.state`, native velocities and unclipped measurements are excluded from predictor inputs, collection decisions and training labels. Any privileged reference is separately labeled.

The challenge is filtering changing state, not retaining an arbitrary number of symbolic associations. It is possible that 16 recent transitions already suffice. That outcome would close the long-memory rationale rather than motivate a shorter window selected after the fact.

## One bounded experiment after collection and runtime are specified

Use one frozen public-trajectory dataset, proposed maximum 256 training, 64 development and 128 final test episodes, each at most 200 native actions. Preserve whole-episode splits and every early termination. Fit the following three models with three paired initializations, identical public targets, episode order and update budget, using final checkpoints only:

| Model | Real information available at each forecast root |
|---|---|
| Persistent action-conditioned GRU | All earlier public observations and issued actions in the episode |
| Trained 16-transition reconstruction | Same GRU tensors/schema, rebuilt from zero from the last 17 observations and 16 intervening actions at every root, during training and evaluation |
| Action-masked persistent GRU | Same modules and work, but past issued-action inputs are zeroed during both training and evaluation |

All three receive the same proposed future action sequence for an open-loop forecast. The action-masked contrast removes actions from history-based state inference, not from the forecast transition. Do not apply a deployment-only reset to inherited weights. No card checkpoint, fitted card readout or calibration scalar transfers into this experiment.

Train on public observation prediction at horizons 1, 5 and 10, with a shared simple readout and the same coordinate scaling fixed from training data. Forecasts may advance private state with supplied actions; they may not assimilate future observations. Use observed public targets, not recovered clean positions. Because clipping makes the observation noise non-Gaussian, squared prediction error measures public conditional-mean prediction, not calibrated state uncertainty. Do not subtract a Gaussian noise floor or relabel this as true-state accuracy.

**Essential collection constraint:** scored future action blocks must be fixed before their forecast root, not chosen by a feedback controller after seeing future observations. Otherwise future actions themselves can convey future measurement information. The collector and its full action-block rule must be fixed before training; uniform precommitted commands are an honest initial option but may yield short episodes. Do not omit early failures or silently resample until enough long histories survive. If the native trajectories do not support the declared long-history comparison, record this candidate as inconclusive. Collector adequacy and an engineering runtime cap remain unresolved, so this memo is not launch-ready.

Add three inexpensive or conventional references to the same saved roots: hold the latest public observation, a fixed 16-transition ridge forecast with actions, and a supplied-physics nonlinear filter with an explicit clipping-aware observation model. Fix any ridge/filter settings using training/development data only. The physics reference may know the published equations and noise law, but never the true current state or future measurement. It is a useful stronger information assumption, not an equally learned model. A raw Gaussian filter that silently ignores clipping is not a convincing reference here.

Report equal-episode normalized squared forecast errors for all three horizons, all three fit pairs and early/late roots separately; include the exact eligible episode/root denominators. The primary comparison uses roots with at least 16 completed actions and ten actually observed subsequent steps, shared across all methods. Also report terminal prediction/coverage separately so surviving horizons cannot be mistaken for whole-task competence. No native return improvement or planning advantage follows from this offline outcome.

## Decision and stopping rule

Before data generation, freeze these proposed practical margins or explicitly replace them in the protocol. They are decision rules, not statistical significance claims:

- No interpretation of long-history value unless at least half the final test episodes contribute a primary root. Failure closes this dataset recipe as inconclusive; do not lengthen episodes or replace failed cases.
- Persistent GRU must reduce primary ten-step error by at least 5% versus both the trained 16-transition reconstruction and the action-masked persistent model, with a reduction in every paired fit. Otherwise stop the persistent action-memory claim on this task.
- If a cheap history predictor is within 5% of persistent GRU's primary error, there is no demonstrated need for a more elaborate memory cell. If the supplied-physics filter is within 5% at lower measured cost, report that a conventional filter suffices when dynamics are supplied. Its extra model knowledge prevents claiming that learned dynamics have no value in general.
- A pass only establishes a forecasting deficit under the specified collector. It does not authorize another large six-family sweep. First state a mechanism hypothesis tied to the remaining error and specify how improved inference should change decisions. A later, separately frozen closed-loop comparison is required for that claim.

Charge real-boundary reconstruction, prediction, copies, retained state, fitting and whole inference time. Equal parameters or forecast counts do not establish equal compute. Report the simple controls even if they win. Do not add a temperature sweep, covariance-inflation arm or a new biological label to rescue an unsuccessful comparison.

**Recommendation:** keep this as one candidate screen, not the next main study. Current evidence supports careful evaluation of conventional action-conditioned filtering; it does not yet identify a novel architecture problem that the existing table, finite-history or supplied-physics controls cannot handle.
