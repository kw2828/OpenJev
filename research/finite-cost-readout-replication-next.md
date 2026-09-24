# Next stability test: factorized recurrent dynamics with a learned readout

**Proposal only, not registered or executed.** The
[fresh replication](finite-cost-readout-replication-results.md) fails its
continuation rule. Its longer-horizon training follow-up remains unadmitted.
The relative readout gain persists, so retain that head in a new stability
diagnostic instead of selecting the one strong seed or rerunning the failure.

## A smaller dynamics family and an initially identical control

The [finite-world source](../src/openjev/research/finite_observation_world.py)
uses action-conditioned transitions, hazard depending on action and next state,
and ordinary observations depending only on next state. The reset odor uses
the same emission law without transition or hazard. Test this explicit
conditional structure with learned parameters:

`B[a,o,n,s] = O[o,n] * (1 - h[a,n]) * T[a,n,s]`

`found[a,s] = sum_n h[a,n] * T[a,n,s]`

Here T is column-stochastic over next states, O is column-stochastic over
ordinary observations, and each hazard h is between zero and one. The blind
survival operator is the sum of ordinary branches. This is a structural prior
taken from the task, not discovery of that structure. The true transition,
emission and hazard values must not be copied into a learner.

| Arm | Dynamics and reset | Trainable parameters |
| --- | --- | ---: |
| Factorized | Learned T, O and h; share O between reset and later observations | 352 |
| Function-matched unrestricted | Existing free branch/reset model, initialized from the factorized model's B, found and O | 1,120 |
| Existing unrestricted | Existing independent dense branch/reset initialization | 1,120 |

Every arm keeps the current learned bounded cost head, initialized identically.
The smaller model has 256 transition, 32 emission, 32 hazard and 32 readout
logits. World-aligned readout initialization remains a privilege. The true
dynamics fit the factorized family at finite logits; the exact cost vertices
remain only limiting values of the shared bounded head.

## Specify the initial functions before training

Use local seeded float64 CPU draws: `T = softmax_next(0.05 * Z_T)` of shape
4 by 8 by 8, `O = softmax_odor(0.05 * Z_O)` of shape 4 by 8, and
`h = sigmoid(-log(32) + 0.05 * Z_h)` of shape 4 by 8. The nominal hazard
1/33 comes from the existing flat 33-outcome initializer, not the world's
true hazard. Reuse the existing transition/reset seed domains and freeze a
third hazard domain, proposed as `fit_seed XOR 0xC2B2AE35`, before qualification.

For the matched unrestricted arm, set branch logits to
`log(concat(B.reshape(4,32,8), found[:,None,:]))` and reset logits to `log(O)`.
This preserves the existing branch layout. Qualify initial prefix states,
event distributions, blind/observed costs and survival within 1e-12 on public
engineering cases. Test found absorption, linear readout expectation and
probability normalization. Gradients and updates need not match across these
different parameterizations. Preserve initial functions and hashes in receipts.

## One bounded comparison

Use nine fits, three fresh seeds per arm, on a common fresh TRAIN/DEV attempt
pool. Keep the existing H2 endpoint objective, H8 evaluation, 512/128 attempts,
480 epochs, batch size 64, Adam 0.003, global clip 5 and coefficient-one prefix
NLL. Pair batch orders and learned-head initialization. Retain terminal events
and global denominators. All nine final checkpoints must precede DEV generation.
No warm starts, true state labels, sticky-transition initializer, RL update,
loss-weight sweep or longer training horizon belongs in this comparison.

Freeze source, seeds, runtime, work budget and all three absolute criteria
before generation. A candidate that fails any criterion cannot advance based
on a favorable mean or isolated seed. Report every paired H4/H8 regret,
blind/observed cost error, event KL, likelihood, parameter/storage count and
actual train/inference cost against both controls. Equal epochs do not mean
equal computation; smaller parameter count is not measured speedup.

If only the factorized model passes, the evidence concerns the combined
structural constraint, parameterization and reduced capacity. If both
initially matched models pass while the existing initializer fails,
initialization remains a plausible explanation. The shared reset/emission
constraint is bundled with factorization and would require a separate ablation
to isolate. Neither outcome proves latent-state recovery.

The earlier [14-coordinate retention comparison](finite-observation-learning-results.md)
already failed, so this is not another claim that sticky initialization solves
learning. Dense transitions can still contract history in the proposed model.
Factorized hidden-state filtering is established modeling; success would give
OpenJev a stronger compact baseline, not by itself an ICLR novelty claim.
Scenario shifts and a second environment remain necessary before transfer or
robustness claims. The failed replication and its original stop rule stay closed.
