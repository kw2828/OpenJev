# Pose adaptation failure: stale support and compounded rollout error

September 19, 2026. This is a post hoc diagnosis of the completed, exposed-data
`pose-adaptation-v1` experiment. Its failed **1/17 requirements, 185/661
comparisons** result is unchanged. All calculations below use saved predictions,
targets and measured public trajectories. No weights were loaded, predictions
regenerated, models fitted, random draws made or simulators called.

**The harmful update is already visible at the first forecast step, then grows.**
Much of the aggregate damage comes from the first window of each source
trajectory: its older support transitions contain unusually large motion
changes that have subsided by the forecast root. This supports testing stale
support and rollout feedback separately. It does not yet establish why the
learned features extrapolate poorly or justify a learned trust mechanism.

## Same weights, different context update

`meta` and `meta_prior` use exactly the same final checkpoint for each of seeds
1101, 1202 and 1303. Only the context solve is enabled or disabled. Each entry
below is **adapted / prior-only RMSE**, pooled over three fits and 160 windows
before taking the square root. Horizons are nominal.

| Panel and endpoint | 20 ms | 100 ms | 200 ms | 500 ms |
| --- | ---: | ---: | ---: | ---: |
| Plain position, m | .000783 / .000507 | .006429 / .004060 | .024102 / .013424 | .223226 / .066646 |
| Plain rotation, rad | .001972 / .001316 | .010984 / .005963 | .026834 / .016740 | .159004 / .086911 |
| Zigzag position, m | .001030 / .000874 | .016312 / .009137 | .077374 / .030169 | .592663 / .145313 |
| Zigzag rotation, rad | .002455 / .002323 | .020419 / .019145 | .065440 / .056031 | .364015 / .251594 |

Adapted pooled MSE is worse at **all 25 horizons on all four endpoints**.
All three seeds are worse at each displayed horizon. Pooled over all horizons,
it is worse on all ten parents for plain position, plain rotation and zigzag
position, and nine of ten for zigzag rotation. Thus neither one seed nor one
parent explains the aggregate failure.

The direct separation between adapted and prior-only forecasts also expands.
Position separation RMS rises from .000664 to .228859 m on plain, and .000931
to .577139 m on zigzag, between steps 1 and 25. Angular separation rises from
.001644 to .174128 rad and .001405 to .310083 rad. At step 1, both branches
query the same measured root and action; no divergence of imagined state has
occurred yet. Later separation combines the weight change with subsequent
state-dependent queries, so it cannot by itself identify feedback instability.

## Broad parent coverage, concentrated window damage

There are 480 fit-window pairs per panel, representing 160 windows evaluated
with three checkpoints, not 480 independent trajectories. The ten windows
starting at raw index zero contribute just 30 of those pairs (6.25%). Define
positive excess as `max(mean_h(error_meta² - error_prior²), 0)` separately for
each physical endpoint.

| Panel | Start-zero share of positive position excess | Start-zero share of positive rotation excess |
| --- | ---: | ---: |
| Plain | 98.51% | 89.81% |
| Zigzag | 97.57% | 53.67% |

This reconciles the parent-level consistency with a concentrated failure: each
parent has a start-zero window. It is a diagnosis, **not permission to exclude
those windows**. On the other 150 windows, pooled all-horizon adapted/prior
RMSE is .02577/.03331 m and .03119/.04338 rad on plain; .06918/.07411 m and
.14277/.12984 rad on zigzag. Zigzag rotation therefore retains a broader
weakness even outside the early-window concentration.

Measured motion supplies a possible causal warning before any forecast target
is available. Using the exact support target definition, let
`y_t = R_t^T(dp_(t+1)-dp_t)` for position, or
`R_t^T(w_(t+1)-w_t)` for rotation, with spatial increments in the same world
frame before subtraction. The table reports vector-norm RMS over support
transitions t=1..30, its most recent five transitions t=26..30, and the actual
future transitions t=31..55. These are unscaled changes per nominal step,
not SI accelerations or the model's component-pooled normalization scalars.

| Start-zero windows | Endpoint | All 30 supports | Latest 5 supports | Actual future 25 |
| --- | --- | ---: | ---: | ---: |
| Training, 30 windows | Position, m/step² | .000600 | .000678 | .000779 |
| Training, 30 windows | Rotation, rad/step² | .001752 | .001981 | .001573 |
| Plain, 10 windows | Position, m/step² | .006618 | .000493 | .000592 |
| Plain, 10 windows | Rotation, rad/step² | .017642 | .001264 | .001149 |
| Zigzag, 10 windows | Position, m/step² | .007881 | .000549 | .000708 |
| Zigzag, 10 windows | Rotation, rad/step² | .011750 | .002088 | .002005 |

The exposed panels' older support changes are much larger than both their
latest support and forecast changes. Later windows have similar overall
support/future RMS: plain .000541/.000567 m and .001294/.001292 rad; zigzag
.000922/.000917 m and .002217/.002262 rad. The old training archive's known
cropping and the longer archives' differing beginnings make a transient
distribution mismatch plausible, but the original generator state is not
authenticated. Calling this a particular reset, collision or hidden-parameter
change would exceed the evidence. The regression already conditions on state
and action; target magnitude alone does not prove a wrong conditional law.

## Oracle selector headroom, not a deployable gate

For each fit-window pair, choose the saved branch with lower **full 25-step
joint loss**: mean position squared error divided by .1² plus mean rotation
squared error divided by .1². Ties use the prior. This oracle sees future
targets; it is only a ceiling for that particular joint-loss binary choice.

| Panel | Joint oracle chooses meta | Prior-only position / rotation RMSE | Joint-oracle position / rotation RMSE | Improvement versus prior |
| --- | ---: | ---: | ---: | ---: |
| Plain | 353/480 (73.54%) | .03288 m / .04214 rad | .02640 m / .02016 rad | 19.70% / 52.16% |
| Zigzag | 226/480 (47.08%) | .07247 m / .12657 rad | .07060 m / .09861 rad | 2.58% / 22.09% |

All three seeds improve on both endpoints under this oracle. However, its
zigzag position error worsens on five of ten parents; the joint objective
allows position/rotation tradeoffs. The 2.58% pooled position gain is not an
upper bound for every alternative multiobjective selector, nor evidence that
a causal gate can reproduce these choices.

Separately selecting the lower-error branch **for each endpoint** gives an
unattainable simultaneous upper bound on endpoint improvements: plain
30.26% position / 53.96% rotation, and zigzag 25.88% / 23.01%. Corresponding
RMSEs are .02293 m / .01940 rad and .05371 m / .09744 rad. Different endpoint
choices can require different branches for the same window. Adaptation has
lower all-horizon endpoint MSE on 297/480 plain-position, 338/480
plain-rotation, 257/480 zigzag-position and 207/480 zigzag-rotation pairs.
There is useful complementary prediction behavior, not merely uniform harm,
but these future-informed counts do not validate a trust rule.

## A decisive bounded probe before a new architecture

Keep the three existing meta checkpoints frozen and retain every window.
Compare the prior, existing 30-support posterior and one predeclared
recent-support posterior (the last five completed transitions,
with unchanged unit ridge). Report the changed support count and effective
prior strength rather than presenting that control as a pure recency change.
No fit, search over support lengths, source-index rule or threshold selected
from forecast targets is needed.

A small recency-by-robustness comparison is more informative than adding
another learned gate now: fixed exponential age weights with half-life five,
per-output Huber IRLS with three updates and delta 1.5 in the existing
normalized target units, and their combination. Center every ridge penalty
at the same learned prior. Initialize IRLS at that prior; each output needs
its own weighted solve. State the age-weight normalization and resulting
effective prior strength explicitly. Recency can suppress stale but
internally consistent support; bounded residual influence can suppress
isolated large innovations. Their interaction helps distinguish these
explanations, although neither identifies a physical cause. Per-output Huber
already limits diagonal residual influence, so another cap would be mostly
redundant. A cap on the entire posterior correction could succeed simply by
turning adaptation off and would need comparison with the prior-only branch.

For each fixed posterior, distinguish two evaluations: the existing free
25-step rollout, and a clearly privileged diagnostic that queries one-step
increment changes at each actual future pose/motion. The latter does not
update weights and cannot be presented as an available online predictor.
Persistent error on true-state queries indicates that the fitted correction
itself does not transfer; good true-state corrections with bad free rollouts
instead implicate accumulated state/query error. Step 1 already establishes
an immediate correction problem for the current full-support posterior.

Before considering a learned gate, test a cheap causal validation signal:
fit earlier context transitions and compare its one-step residuals with the
prior on the most recent held-out context transitions. No future pose, parent
identity or window-start index may enter the decision. Keep the all-prior,
all-adapted and fixed recent-support controls. A signal that catches only
obvious early transients would support a simple safeguard, not a new memory
architecture. This probe is proposed only; nothing above executes it or
changes the failed study's criteria.
Retain the old static, GRU and classical controls in any reported comparison;
improving on the damaged full-support posterior alone is insufficient.

## Authentication and limits

The published manifest bound the protocol, completion and audit files. All
149 producer payload hashes/sizes, twelve frozen source hashes and seven
prepared-data hashes were checked before calculation. Saved pose errors were
independently recomputed in NumPy and matched the audit arrays. The final
source check still matched. The producer/audit manifests bind the individual
prediction and target files; no checkpoint deserialization was used.

| Artifact | SHA-256 |
| --- | --- |
| Protocol | `f029b98d96db6fe3aa0383290b22a71f3ae53305a1b5e506c5b37e48d2056615` |
| Producer completion | `c7cc39977ee5dade2cf610aa7bb795e7f9eb34fad97b337c723437200988c799` |
| Audit receipt | `74e3ed560fce14569912dfbbee32603adb932c7e2efa90b63be6695c057df5d8` |
| Audit summary | `fa9c973f2f3185dae3575f16b4679043724896d2625dd7fe4b3792d007d88473` |
| Saved error arrays | `69abb5c4f491c9fc26156670d938a0b0b9da174b8038c7513feab8dd4ae654b8` |
| Training arrays | `f6b049365f931d1ead0806f53883ae35a5fccb63b5e6fd75df5d782833bfe9b4` |
| Original development arrays | `30d259ef22f231e271a0c44c4fbfe0b25a01ad3ad576c67df91731c9f665f563` |
| Plain arrays | `1a402646581edbd6152c51db090f401e357c744db9d2e8c10ce2a37ec0f077f6` |
| Zigzag arrays | `7ad0ad932c480add80952ad22a176039f6ffa1319106f6d3a3d0b7abf5373407` |
| Normalization | `0e07ba075c29987455cedb11971b95ce92f9509bee99b2795439ab4daf11beec` |

Measured support geometry was reconstructed in float64 from unnormalized
public observations using the inferred RzRyRx convention. Angular increments
used the principal SO(3) logarithm; these adjacent rotations were away from
its cut. The source timing is nominal, future inputs are recorded applied
torques, and the data are simulated and exposed. Windows and seeds are not
independent samples. These descriptive partitions and oracle calculations
provide neither significance tests nor a causal explanation, fresh evaluation,
calibrated uncertainty or new efficacy claim. Raw data and forecasts remain local.
