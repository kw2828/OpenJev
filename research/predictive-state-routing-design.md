# Predictive-state correction: implementation and a failed benchmark match

**The subsequent [multivariate pilot](fsm-correction-results.md) fails its rule,
5/9 conditions pass.** All 30 fits complete, but selective correction is less
accurate and slower than dense correction. The registered routing candidate
stops here; no architectural novelty or efficacy is established.

The initial [fabricated qualification](predictive-state-qualification.md)
implemented the correction core and ParWH adapter without measured-data access.
It found that a single-output benchmark cannot test adaptive routing with this
equation. The later multivariate experiment addresses that limitation and still
finds no advantage. The original derivation and design rationale follow.

The [joint-observer study](robot-joint-observer-results.md) still fails its
development rule. Longer conditioning did not beat the strongest historical
control. Neither this work nor the earlier connectivity comparisons establish
a biological wiring advantage. The previously proposed observer transfer
remains unlaunched.

## Implemented mechanism

[PredictiveStateCorrection](../src/openjev/research/predictive_state_correction.py)
has three equal state blocks, a GRU transition, a linear observation decoder
`C z + b`, and explicit caller-owned state. Every block advances through the
transition. When an observation arrives, the model computes:

```text
prior = GRU(input, previous_state)
error = observation - (C prior + b)
gradient = error @ C
block = argmax_m sum(gradient[m] ** 2)
state = prior + mask * gradient / (sum(C ** 2) + 1e-6)
```

Lowest-index ties are deterministic. Dense correction updates every coordinate.
Selective correction updates the largest-gradient block. The rewired control
updates its cyclic successor, using that destination block's own gradient.
All modes have exactly the same parameters, initialization and auxiliary heads.
The transition is dense: unselected coordinates are preserved by the correction
itself, but subsequent transitions can mix information across all blocks.

Three training-only heads read their corresponding state blocks and the logged
input prefixes of lengths 1, 8 and 32 to predict their endpoint observations.
Their targets must come from those executed inputs. No counterfactual outcomes
are supplied by ordinary trajectory data. No target enters online correction;
autonomous rollout accepts future inputs and state only.

For fixed `C`, the step size is at most `1 / ||C||_2^2`. Thus any of these
coordinate masks gives a nonincreasing current squared observation residual
in exact arithmetic. This says nothing about subsequent forecast accuracy,
global recurrent stability, probability calibration or closed-loop control.

## Why ParWH cannot establish adaptive routing

For a single output, `error` is a scalar. Each block score is
`error**2 * sum(C[m]**2)`. With frozen weights, every nonzero residual therefore
chooses the same block, regardless of observation, input or residual sign.
Zero residual gives zero correction. A rank-one multivariate decoder has the
same limitation away from zero projected residual.

Consequently, a win on the single-output Parallel Wiener-Hammerstein circuit
could support fixed block correction, but **could not support adaptive routing**.
Its [adapter contract](parwh-data-contract.md) remains useful for a future
identification comparison. It is not admitted as the primary test of this
mechanism. This is an analytic limitation, not a failed empirical fit.

The corrected question is whether selective correction improves later forecasts
when innovations contain multiple independent directions. Multivariate output
alone is insufficient: the learned decoder and observed residuals must actually
produce different selections. A future comparison must include a fixed-block
control in addition to dense and rewired correction and a conventional recurrent
predictor. Route occupancy and unchanged-component forecast error are diagnostics,
not substitutes for an overall utility-versus-compute improvement.

## Prior art and candidate benchmark

[RIMs](https://arxiv.org/abs/1909.10893) already selects recurrent modules;
[dynamic predictive coding](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011801)
already uses prediction errors for state correction;
[KalmanNet](https://arxiv.org/abs/2107.10043) learns innovation-dependent gains.
[Predictive-State Decoders](https://proceedings.neurips.cc/paper_files/paper/2017/file/61b4a64be663682e8cb037d9719ad8cd-Paper.pdf)
supplies auxiliary prediction targets, and
[RPSP](https://arxiv.org/abs/1803.01489) conditions predictive representations on
future actions. Decoder-tied block selection is recognizable block-coordinate
predictive inference. These ingredients do not establish a novel architecture.

The [CubeSpec Fine Steering Mirror](https://github.com/merijnfloren/fsm-benchmark-data)
is a possible multivariate benchmark: three applied voltages and three measured
displacements. Its authors provide linear state-space and nonlinear LFR references.
Those controls matter because the platform is mostly linear. The subsequent
registered study uses a custom subset of its estimation data; its official-test
and 300 mV arrays remain undecoded.

## Original pre-comparison requirements

This checklist preceded the subsequent [frozen protocol](fsm-correction-protocol.md).

Freeze an appropriate multivariate dataset, input timing, whole-record splits,
model/control rosters, loss, fit budget, seeds and stopping rule. Keep predictive
supervision and available information matched. Measure complete request latency
including routing and correction preparation, and report model/state storage.
Compare to a strong identification reference, not only a small weak GRU.

The initial proposed development margins were at least 5% lower equal-record forecast
error, no record more than 2% worse, and at most 10% higher latency or persistent
numeric storage than the declared matched control. These were **not yet a
registered study** at qualification; the later protocol fixed the full comparator
and rule. No ICLR-readiness,
control performance or biological interpretation follows from qualification.
