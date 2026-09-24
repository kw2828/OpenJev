# Expected-count recurrent learning: arithmetic qualification

Prospective engineering prerequisite, **finite-expected-count-qualification-v1**.
This implements the first stage of the [next learning proposal](finite-gap-readout-study-next.md).
It does not reopen any stopped scientific study or generate new benchmark data.

## Mechanism and information boundary

Use the same categorical transition T[action,next,current], shared observation
law O[ordinary observation,next], and hazard h[action,next]. Reset observes O
from a uniform prior, without a transition or hazard. An action transitions
first, then either terminates with hazard h or produces O after survival.
The terminal found event is index 4; it has no ordinary emission and permits no
subsequent event. The hidden terminal destination is retained only to calculate
expected counts, not as a live policy state.

Scaled forward-backward inference returns sequence likelihoods and expected
transition, emission, survival, found and reset counts. Reset counts contribute
to O exactly once. Sequence log likelihoods are summed; they are not averaged
per sequence length. State smoothing may use later TRAIN observations.
Forward filtering at deployment remains causal. The kernel accepts public
observations and actions only, with no true hidden-state or oracle-posterior
input. A strict adapter extracts labels from the existing 9 by 31 public tokens.

Plain maximum-likelihood updates permit boundary probabilities and retain old
values only at exactly zero exposure. The Torch bridge instead uses an explicit
MAP update with pseudocount **0.001** for every T/O category and each of the two
hazard outcomes. This equals adding 0.001 times the sum of log probabilities to
the sequence log likelihood. The gradient control optimizes that same objective,
divided by the fixed number of valid public events. It uses full-batch Adam at
0.003 and gradient norm clipping at 5. The head receives no gradients or updates.

Both updates must preserve the model's probability meaning when converting
between probabilities and finite logits. No clipping, hidden-state alignment,
or silent probability repair is permitted. MLE and MAP are distinguished; a
monotonicity check for the penalized prefix objective supplies no guarantee for
the later joint decision objective.

## Independent checks and retained witnesses

An independently authored oracle enumerates all hidden paths using rational
arithmetic with two states and asymmetric action matrices. Seven fixed action
and observation sequences cover reset-only, ordinary and terminal histories
through three actions. Check likelihood, scaling factors, filtered and smoothed
states, transition posteriors and every expected count. The retained witnesses
require absolute error at most **2e-12**. Tests additionally check stronger
component tolerances, count conservation, impossible evidence, padding,
ownership, zero-exposure updates, and nondecreasing MLE/MAP objectives on these
fabricated cases. Change only a future observation and confirm earlier filtered
states are unchanged while smoothed states may change.

The real eight-state Torch model is initialized with engineering seed **937101**.
Eight hand-written public histories comprise four full eight-action histories
and four first-found histories of lengths 2, 4, 6 and 8 including reset. No
world sampler, stored training data or old checkpoint is consulted. Check:

- Torch likelihood and prior against the separate NumPy computation.
- Three representative dynamics gradients against finite differences of the
  NumPy likelihood, covering transition, emission and hazard parameters.
- Three EM and three gradient updates, retaining every objective and final
  parameter array. Cost-head bytes must remain unchanged.
- Strict probability/logit round trips and identical paired initial operators.

EM and gradient update counts are equal here for API qualification, **not compute
matching**. Record all inference, diagnostic and gradient passes and elapsed
time. The witnessed learning traces are engineering checks, not generalization
results. They are not a method-selection or hyperparameter-selection rule.

## Execution and closure

Before running tests or numerical witnesses, register this protocol, every local
source in the import chain, tests, pinned dependency versions, current runtime,
and the fixed fixture/method roster. One CPU thread, no accelerator or external
model. The original suspend-inclusive supervisor allows **600 seconds**, 4 GiB
worker RSS and 64 MiB of phase outputs. All output paths are exclusive. Success
requires lint, all selected tests, all witnesses, unchanged source hashes and a
successful original process closure. Preserve any failed attempt; a repair
requires a separate source version and registration.

This is an established inference and learning baseline, not a novel recurrent
architecture, connectome mechanism, calibrated policy or performance result.
A scientific comparison remains separate: it must freeze fresh data and seeds,
the unpretrained/gradient/EM arm roster, total compute accounting and the joint
training schedule before generating empirical arrays. No scientific successor
is admitted by a passing engineering test alone.
