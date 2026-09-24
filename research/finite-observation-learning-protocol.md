# Finite observation learning diagnostic v1

Prospective protocol. This is a deliberately favorable, exactly specified finite
world, not a robotics, OTTO, chess, language, connectome or novelty benchmark.
It tests whether the new observation-operator component can learn decision costs
and retain useful distinctions over longer unobserved intervals. The previous
[conditional-label experiment](otto-conditional-label-results.md) remains closed
as DEV_FAIL. This diagnostic does not override its continuation rule.

## World and information boundary

There are eight hidden states, four actions, four ordinary observations and an
absorbing found event. The four deterministic transitions are xor1, increment
modulo8, three-bit left rotation, and xor4. Transition noise mixes each with a
uniform state at weight0.02. After transition, found probability is
0.005+0.005*(bit2(next) xor(action mod2)). Otherwise the ordinary observation is
the lower two state bits with probability1-epsilon, or each other symbol with
epsilon/3. Epsilon is0.12 for TRAIN/base DEV and0.30 for shifted DEV.

The terminal decision has zero cost for ((state xor(state>>1)) mod4), otherwise
one. Four costs are centered across decisions. These are classification costs,
not native planning returns. Targets are analytic finite-world probabilities
up to float64 roundoff, conditional on sampled contexts; categorical random
sampling is not claimed to implement a bit-exact rational law.

Each attempted case starts uniformly, observes an initial symbol, then takes
eight random actions. Cases found during that prefix are excluded without
replacement. The model receives nine31-wide float32 tokens containing only
action and observation indicators and an initial reset marker. Remaining token
slots are zero. It receives neither the oracle belief nor epsilon. Forecast
actions are sampled before future observations. Forecasts and their targets
precede the current observation. Blind rollouts ignore all future observations
and retain unnormalized surviving mass; observed found suffixes are absorbing.

## Five arms and paired training

Three fit seeds420261001,420261002,420261003 each train all five arms:

| Arm | Transition | Initialization | Parameters |
| --- | --- | --- | ---: |
| tied_dense | Blind operator sums observation branches | Existing dense |8778|
| untied_dense | Independent blind operator | Existing dense |9618|
| tied_retentive | Blind operator sums observation branches | Retentive |8778|
| untied_retentive | Independent blind operator | Retentive |9618|
| gru | Ordinary gated recurrent updates | Seeded PyTorch |11181|

The operator models have14float64 latent coordinates, a float32 prefix GRU,
and a linear centered readout. Retentive initialization uses0.95I+0.05U,
nominal found probability1/57, and four equal observation branches, with the
original small logit noise added. Coupled normalization also changes actual
hazards, so this is an initialization bundle, not an isolated retention effect.
Tied/untied arms share initial observed operators, encoder and readout within
each initialization. Untied blind logits initially marginalize the same raw
observed logits. The new GRU is not the previous8299-parameter GRU.

Study RNG namespace420260924; engineering namespace919001. Each attempted case
uses PCG64 SeedSequence([namespace,split_id,attempt_index]). TRAIN split0 has512
attempts and horizon2. DEV splits1/2 have128attempts each and horizon8.
No replacement cases. Minimum retained supports are256TRAIN and64per DEV.

Every fit uses48epochs, batch64, Adam(lr0.003, default betas/epsilon), norm clip5,
no schedule, early stopping or checkpoint selection. A shared permutation per
seed/epoch uses PCG64 SeedSequence([fit_seed,epoch,818]). Rotate arm execution
order by seed index. The loss sums blind centered-cost MSE, observed centered-
cost MSE, half the sum of blind/observed survival MSE, and full five-event soft
cross-entropy. Cost scale is1. No DEV normalization. All15final checkpoints
must exist before any DEV cases are generated or decoded.

## Metrics and fixed gates

Report every arm, fit seed, regime and horizon1,2,4,8 separately: blind cost
MSE and decision regret, blind survival MAE, observed cost MSE and survival MAE,
full observation-law KL, and blind regret with a fixed cyclic prefix shuffle.
The shuffle moves the complete prefix and length together within each regime,
offset1, holding forecast actions/targets fixed. It is descriptive sensitivity
evidence, not a significance test. Positive target mass with zero predicted
support is an explicit numerical failure, never clipped.

Decision regret is true_cost[argmin(prediction)]-min(true_cost). Learned costs
use raw argmin. A separate privileged known-dynamics uniform-state reference
starts from1/8 after the prefix and propagates the exact blind operator. It is
not the optimal history-ignorant predictor because prefix retention conditions
the state distribution. For that reference only, break numerical ties by the
lowest index within1e-12 of the minimum. Exact rational symmetry motivates this
rule; it is frozen before any empirical data generation.

BASE_TRAINABLE requires every tied_retentive fit seed to satisfy all of:

- At each ofH4 andH8, blind MSE and regret at most half their strictly positive
  corresponding uniform-state reference values.
- H8 blind survival MAE at most0.05.
- At each ofH1 andH2, observed KL at most0.1.
- Minimum supports stated above.

SHIFT_TRANSFER is separate: the same blind H4/H8, survival and support checks
on the shifted regime. Observed shifted KL is descriptive: the model is not
told the emission law changed. Failure to predict an unannounced new sensor
law is not automatically failure to learn the original world. Neither outcome
automatically admits a native experiment or an architecture-superiority claim.
All five arms remain in the report regardless of the two named gates.

## Execution and reporting

Source hashes, runtime, configuration, exact output paths and qualification
receipt are registered before the first study-namespace data call. A native
suspend-inclusive supervisor bounds qualification to300s, fit+evaluation to
1800s, and independent saved-output audit to600s. Maximum worker RSS is4GiB,
maximum phase output512MiB. Single CPU threads; no accelerator, network model
calls, paid inference or new datasets. Preserve failed attempts; no retries,
replacement seeds or cap extensions under the same registration.

The independent audit reconstructs targets from public histories using its own
rational coefficients and scalar arithmetic, checks saved predictions and
metrics, and computes both gates. It does not rerun models or optimize anything.
Close the original producer before audit. Report actual label construction,
training and evaluation time, parameter storage and all final checkpoints.
Same updates or carry bytes do not imply matched capacity or total compute.
If learning fails even here, diagnose representation/training before adding
RL, conformal wrappers or a more complex native environment.
