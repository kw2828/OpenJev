# Query-written memory with aligned predictions and updates

Status: new engineering work after the [protected-readout comparison](otto-protected-readout-results.md)
failed. No new scientific collection, fitting, held-out evaluation or autonomous
run is admitted by this note. The previous experiment remains closed.

## Question

Can a separate memory learn useful score corrections during an episode, using
only genuine query answers? Does a history-conditioned key help beyond the
current key and a simple remembered error?

The failed candidate learned a static action residual offline. This mechanism
instead updates runtime memory when a teacher answer actually arrives. It
retains the same slow recurrent predictor and exact query outputs. Joint AUX
must remain an ordinary-model control, given its stronger descriptive result.
Any scientific comparison needs freshly trained predictors and fresh cases;
the exposed VALID set cannot tune this design.

## Align the target with the representation

Let `C(v) = v - mean(v)` across the four action scores. Let `b_t^-` denote the
complete slow prediction before the current query answer is assimilated,
including its previously learned action residual. Let `M` be an episode-local
matrix and `z_t` the causal key used for both reading and writing.

At a later actual query, define:

```text
r_t = C((Q_t - b_t^-) / 64)
u_t = C(M^- z_t)
M^+ = M^- + eta * (r_t - u_t) z_t^T / (epsilon + ||z_t||^2)
```

For a fixed key and target, this is a normalized squared-error gradient update.
With `0 <= eta <= 1`, its effective correction fraction is at most one in exact
arithmetic. This does not guarantee stability or improvement across changing
keys, targets and subsequent decisions. Zero keys cannot learn a correction.
Nonfinite numerical state must fail visibly, not be silently replaced.

At a nonquery step, return `b_t + 64 * C(M^- z_t)`. At a query, return `Q_t`
exactly. The first query has no prior-error target and performs no write.
Optional memory decay happens before reading; only actual later queries write.
Memory never feeds the slow predictor, its prior, its correction input or carry.

Two tempting shortcuts are invalid here. Using a current-key error with a
different historical write key is not the gradient of this stated objective.
Using the old base-only prior while adding corrections to the full action
forecast can count the existing action residual twice.

## Instantaneous and history-conditioned keys

Normalize each supplied causal key `k_t` by `norm(k_t).clamp_min(epsilon)`;
use the same rule for the mixed trace. Zero or cancelling traces stay zero,
with ordinary differentiation through the clamped denominator. There is no
fallback to a different key. Maintain the bounded feature trace
`e_t = rho * e_(t-1) + (1-rho) * k_t`, starting at zero. The instantaneous
control uses `k_t`; the history-conditioned model uses normalized `e_t` for
both its read and its query update. Future representation training must be
specified separately; the memory kernel itself owns no optimizer parameters.

Engineering controls cover no memory, decaying last-error correction,
instantaneous delta writes, trace-based delta writes and trace-based additive
writes. A no-write intervention preserves normal reads and decay while
disabling updates. It does not have equal computation to the writing model.
The additive control replaces `r_t - u_t` with `r_t`, retaining the same
denominator. Last-error memory overwrites its decayed state with `r_t` at
later queries. Zero-initialized no-memory and no-write outputs are identical
by construction; they are execution controls, not independent efficacy arms.
Counters record attempted operations, including zero-valued writes. Neither
counter values nor local contraction establish learning or global stability.

A representation-perturbation control cyclically rotates the past trace before
mixing it with the current key, with a shift determined only by the absolute
step. It maintains the undisturbed chronological trace as state. Rotation
preserves the past vector's norm; it need not preserve the mixed cue's norm or
angle before normalization. This control is not an order shuffle and cannot
identify temporal credit assignment by itself.

The kernel exposes a normalized correction before writing at each valid key
step, including queries. Adding 64 times this correction to the shadow prior
allows a future evaluator to score the forecast before seeing the answer.
The final action output still copies actual query answers exactly. Numerical
separation from the slow model does not imply gradient isolation: a future
training protocol must explicitly decide which key encoder can learn through
memory writes and where gradients are detached.

This is **history-conditioned online correction**, not an established
eligibility-gradient or delayed-reward learning method. Fast-weight delta
updates are [prior art](https://proceedings.mlr.press/v139/schlag21a.html).
The repository also already has
[observed-value associative updates](../src/openjev/research/card_associative_memory.py),
[policy fast weights](../src/openjev/research/associative_policy.py) and
[chemical-graph eligibility state](../src/openjev/research/mushroom_body_fast_memory.py).
The new task-specific question is whether correctly aligned query supervision
improves decisions under sparse teacher access.

## A real schedule shift needs a new adapter

Existing predictors, capture hooks, feature clocks and metrics assume a query
every four steps. Keep their sources unchanged. A separate adapter supports
periods four and eight with an explicit schedule identity in its carry.
Inserting a made-up answer at step four would change the state and is forbidden.

The adapter must expose the complete causal shadow prior, its pre-assimilation
hidden key, and each nonquery hidden key. Chunk-end hidden state cannot supply
earlier keys. Period-four outputs must match the old canonical predictor under
identical weights, parameter flags and gradient context. Period-eight behavior
needs an independent scalar-recurrence check, including its reduced number of
query-correction transitions.

Only scheduled answers enter model inputs. Other census labels may be retained
for losses or evaluation by a future harness, with strict input separation.
Changing the forecaster's query schedule on fixed collector paths is an
observation-schedule test. It does not establish autonomous period-eight
behavior or teacher-call savings. Initial/later scopes and age denominators
must also follow the declared schedule; period-four cutoffs cannot be reused.

## Evidence needed before scientific execution

Fabricated checks must establish read/write alignment, gradient paths through
keys and writes, exact query outputs, poisoned-label exclusion, chronology,
episode reset, owned carry, chunk equivalence and unchanged slow forecasts.
Record actual operations and recurrent transitions, not inferred speedups.

Then register fresh training, development, test and delay-shift cases; fixed
training schedules, costs and a single continuation rule; and an independent
saved-output audit. Keep all ordinary controls and the practical unpadded
baseline. A mechanism finding still requires complete autonomous episodes,
total-compute comparisons and confirmation beyond one environment before an
architecture or biological-learning claim. Connectome topology remains a later
matched-control question, not a label for this matrix update.
