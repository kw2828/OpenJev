# Does persistent learned memory improve control?

**Protocol frozen; scored execution pending.** The complete engineering
rehearsal passed with 12 tiny fits, 51 controller rows and 2,850 native
transitions replayed with zero error. The combined implementation suite passed
300 tests. These checks establish working machinery, not scientific performance.
This comparison follows the [failed objective-ablation continuation
rule](reacher-objective-ablation.md). The earlier 5.8% mean latent-objective gain
was inconsistent across fits and did not establish useful memory. This study
tests that explanation before adding another architecture.

## Comparison

| Arm | Information retained between real decisions | Imagined planning |
| --- | --- | --- |
| Persistent residual GRU | Complete learned recurrent history | Same GRU transitions and CEM256 |
| Current-packet GRU | Current packet only, including missingness and age | Same GRU transitions and CEM256 |
| Three-packet GRU | Last three real packets and two issued commands | Same GRU transitions and CEM256 |
| Current-packet MLP | Current packet only | Action-conditioned packet prediction and CEM256 |

The first three models have identical parameter sets and paired initial values.
The MLP has 36,599 parameters versus 36,805 for the GRU64, and an independently
named initialization. Every arm uses the same 768 training episodes, Anchor
sequence loss, 48 epochs and 1,152 optimizer updates per fit. There are three
paired fit identities and 12 fits in total. These are new fits; no prior final
weights are reused.

The current-packet models discard previous state during training as well as
deployment. This avoids treating a post-training state reset as evidence about
a model trained without history. The three-packet model reconstructs its state
from a strict physical-step window. It does not retain its own old predictions
through an observation gap.

All final models must restore with their actual classes, planned seed roles,
orders, public training data and optimizer state before any evaluation draw.
Evaluation uses 64 new paired control cases with full sensing, six-step gaps
and ten-step gaps, plus 96 separate prediction episodes. Every model uses the
same case-specific reset, actuator noise and search innovations. There is no
checkpoint or best-fit selection.

The 51 control rows include five references on each sensing panel: zero action,
uniform action, supplied physics with true state, a supplied-physics particle
filter, and a supplied-physics two-measurement estimator. The last estimator
uses wrapped angular displacement divided by actual elapsed time between valid
observations, then propagates nominal dynamics during gaps. It sees public
packets and issued commands, never true velocity or realized actuator noise.
It is a physical-knowledge competence reference, not a matched learned model.

## Frozen continuation rule

Persistent memory must satisfy all 25 checks:

- At least 5% lower family mean native cost than the trained current-packet GRU
  on both gap panels, with every paired fit strictly better: eight checks.
- At least 3% lower family mean cost than the three-packet GRU on the longer-gap
  panel, with no paired fit worse: four checks.
- No more than 2% worse full-sensing family mean cost than the current-packet
  GRU, and no worse than the MLP family mean on either gap panel: three checks.
- Every persistent fit at least 10% better than zero action on each gap panel:
  six checks.
- Known-state physics and the public-history particle filter each at least
  10% better than zero action on both gap panels: four checks.

Native cost is negative total native reward over the same 50 physical steps.
Lower is better. The kinematic reference is fully reported but is not a primary
continuation check. It helps assess whether conventional estimation explains
the result. Intervals and prediction errors are descriptive; neither can rescue
a failed continuation rule. The [frozen protocol](../evidence/reacher-memory-ablation-v1/protocol/plan.json)
has SHA-256 `05f09ee5425190f5d652aedb10af1641037035d85e906d04d86baf33552a5786`.
It permits one execution with a 3,600-second cap and a separate 300-second
saved-output audit, without retries, resumed fits or cap extensions.

## Cost and interpretation

Equal data and optimizer updates do not match computation. The three-packet
model reconstructs history at every real observation and the MLP has different
arithmetic. Report every fit's wall time, assimilation/reconstruction work,
planning work, state payload, trace storage and native stepping. Shared-host
batch timings are not isolated single-agent latency or full FLOP counts.

A pass would establish useful persistent public-history information under this
training recipe. Retaining the last valid angles could explain such a gain;
the present erased-history controls do not isolate velocity inference. A
separate hold-last-angle comparator would be needed for that mechanism claim.
The gate would not establish a new architecture, useful Bayesian inference,
biological wiring or an ICLR contribution. If the shorter history or simple estimator explains the
gain, report that and avoid expanding the claim. Persistent hidden dynamics
need a separate, identifiable task before a slow-context or plastic-state
mechanism is justified. Any later connectome claim needs matched conventional
and rewired controls, untouched confirmation cases and a second environment.

The new [runner](../scripts/reacher_memory_study.py),
[protocol](../src/openjev/research/reacher_memory_protocol.py) and
[independent audit](../scripts/audit_reacher_memory_study.py) passed a complete
rehearsal using engineering-only namespaces. The [capacity measurement](../evidence/reacher-memory-capacity-v1/README.md)
supports the chosen execution allowance, but is a short synthetic projection,
not a runtime guarantee. No scientific result is implied by those checks.
