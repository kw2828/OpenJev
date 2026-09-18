# Can better search make the existing world model useful?

Status: frozen evaluation-only experiment, prepared September 18, 2026.
No inherited model was scored on these new cohorts before freezing the
[protocol](../evidence/reacher-search-v1/protocol/plan.json).
Its SHA-256 is `8d5da5ee146c95532721fd1b6582bc37a091d5417e5d8b68f73e241379f2428b`.
The [corrected six-fit result](reacher-reward-residual-control.md) improves
prediction much more than control. This comparison isolates search allocation
before changing training, latent representations or biological wiring.

## Mechanism and sources

[PlaNet](https://arxiv.org/abs/1811.04551) uses iterative Gaussian proposal
fitting for latent planning. [iCEM](https://arxiv.org/html/2008.06389v1), section
3, additionally studies correlated actions, elite reuse, boundary clipping,
mean evaluation and selection of an evaluated sequence. We borrow adaptive
Gaussian search and explicitly paid mean evaluation. This is an anchored CEM
adaptation, not a reproduction of either system. No warm start, colored noise,
proposal momentum or cross-decision elite reuse enters the first comparison.

The [search component](../src/openjev/research/reacher_adaptive_search.py)
accepts explicit innovation arrays and a score callback. It does not access a
simulator, model weights or a random generator. This separates proposal
accounting from both model inference and the experiment's stream manifest.
Its synthetic checks cannot establish control effectiveness.

The component passes 54 synthetic checks for exact budgets and call schedules,
paid mean evaluation, global-best retention, input immutability, shared prefixes,
clipping, invalid scores and terminal boundaries. Independent review caught and
corrected an extra random-shooting score call before any scored experiment.
The complete runner and independent auditor add 124 checks, for 178 passing
checks in the combined suite. A random-weight rehearsal reconstructed every
saved proposal and replayed native transitions. These checks establish harness
behavior, not model effectiveness.

## Fixed comparison

| Planner | Scored sequences per decision | Allocation |
|---|---:|---|
| RS64 | 64 | Common bank with the original seven anchors and variance split |
| RS256 | 256 | Same 64 plus 192 independent proposals |
| CEM256 | 256 | Same 64, then three adapted generations of 64 |

Scorer call sizes are exactly `[64]`, `[256]` and `[64, 64, 64, 64]`.
Random shooting scores its full bank in one call, so extra batching overhead
is not quietly charged only to that baseline.

All three arms receive the identical new common bank. It retains the original
anchor values and proposal distribution, not the old run's exact draw assignment.
Innovation arrays always cover the full planning horizon, even near termination;
unused trailing blocks are recorded but never scored. This differs from the
legacy helper's shortened random-array shape and must be declared in the new
stream contract. Additional random-search samples
split evenly between standard deviations .25 and .75. Do not create the common
bank as a prefix of an array whose shape changes with the planner budget:
case/sample RNG assignment and the original variance split would change.

CEM fits a diagonal Gaussian to the best eight candidates of each current
generation, with a standard-deviation floor of .001. Subsequent generations
use 64, 64 and 63 innovations. The last generation's remaining slot evaluates
the current proposal mean. Fixed anchors are evaluated once; elites only fit
the distribution. Select the globally best evaluated sequence, with earliest
candidate ID breaking ties. There is no unpaid final score.

Commands use the existing three-step blocks, clipping and float32 conversion.
The horizon is `min(12, 50 - decision_step)`. Both 256-sequence planners perform
136,704 imagined transitions per 50-step case, versus 34,176 for RS64. These
counts exclude shared observation assimilation and executed-action state
updates, which must also be reported. Proposal generation, sorting, fitting,
clipping, copying and trace storage all enter decision timing.

Evaluate all six unchanged free/residual fits at seeds 271, 283 and 293 on 64
fresh paired cases under full, ordinary and shifted sensing. This is 54 learned
controller/panel rows. Retain all four reference controllers and clearly label
their original 64-sequence budget and supplied-physics access. Keep all reset
interventions required for any memory qualification in a separately explicit
schedule; the planner comparison does not itself earn that qualification.

CEM256 versus RS256 is the primary equal-scoring-budget contrast. RS256 versus
RS64 measures spending more computation. Neither contrast implies equal wall
time. Use all six fits without selecting a best seed or reward head.

## Separate prediction error from search exploitation

[MOPO](https://arxiv.org/abs/2005.13239) treats distribution shift as a central
problem for offline model-based control. Better predicted reward under stronger
search can reflect model error. Test this on common states rather than comparing
optimistic scores from controllers that have reached different states.

The diagnostic uses 16 fresh mixed-policy episodes, with four prespecified roots
per episode: step 6, two steps into the ordinary blackout, the first observation
after that blackout, and step 47. Replay all three sensor histories over the
same physical records. Reconstruct each model's state solely from public
observation packets and issued commands.

At each root, evaluate a common 64-sequence bank and every model/planner's
selected sequence in cloned native states under four shared, independent noise
branches. Retain all 118 identity slots across all sensing panels, including
duplicate sequences. Report the number of unique sequences separately; retaining
duplicates keeps the native evaluation budget fixed and the identity mapping
explicit.
Native results are diagnostic labels only, never selection inputs. Save commands,
noise, integration state, time index, observations and rewards for replay.

Report common-bank rank agreement, native selected return, prediction error,
clipping frequency and regret within the evaluated candidate union. This is
finite-set regret, not regret against an optimal controller. More optimistic
CEM scores combined with worse native selected return would favor a model-error
explanation. It would not justify a stronger-search promotion.

The last root has only three remaining physical transitions. No terminal value,
post-terminal simulator step or imaginary advance of physical time is permitted.
Restore native integration state and the time-limit index; supply the recorded
branch noise explicitly because restoring physics alone does not restore the
wrapper's noise generator.

## Freeze and continuation

Before scored calls, bind all six checkpoint hashes, source, runtime, exact
execution order, fresh cohorts and role-separated generators. Enumerate initial
states against original training and every prior evaluation. Common banks,
actuator noise and bootstrap pairing require named intentional reuse; noise
must never share a candidate generator. Separate random-search extensions,
CEM innovations, diagnostic collection and native diagnostic branches.

The frozen full-run cap is 3,600 seconds, including diagnostics and final
artifact hashing. Before freezing, use separate engineering fixtures to verify
that the required scope fits. No retries, replacement seeds or budget extensions
are allowed after the scored run begins. A failure becomes a preserved attempt.
The independent auditor reads saved predictions/proposals and replays native
transitions; it must not rerun learned candidate scoring.

Planning continuation requires at least 5% lower residual-family mean
cost for CEM256 versus RS256 in both ordinary and shifted panels, with every
paired residual fit improving. Publish free-head controls, all seeds, all costs
and diagnostic failures regardless of outcome.

The scored namespace is `reacher-search-v1-scored`. Before freeze, random-weight
engineering fixtures had exercised the initial draft namespace. The final
protocol excludes that namespace and three separately named engineering
namespaces at full production role coverage. Its 1,277 generator states are
disjoint from 5,108 enumerated engineering states and all three earlier study
namespaces, including original training. No inherited checkpoint was evaluated
during those engineering fixtures. The
[preflight note](../evidence/reacher-search-v1/engineering-namespace-note.json)
preserves this correction.

Carry the earlier failed 17-check result as authenticated historical context.
The new study checks control competence for each planner, but does not rerun the
earlier prediction endpoints or memory-reset interventions. It therefore cannot
claim to have passed the earlier full continuation rule or established memory
use. A search improvement is a conventional planning result. If ranking degrades, study model
exploitation and training coverage. If ranking improves but control remains
weak, next vary physical planning horizon under a new matched comparison.

## Reproduce

Use the pinned [robotics runtime](robotics-requirements.txt) and the complete
upstream artifacts documented in the
[six-fit report](reacher-reward-residual-control.md#costs-and-verification).
The new runner authenticates every inherited weight and source before scoring.

```bash
PYTHONPATH=src:scripts .venv-robotics/bin/python scripts/reacher_search_study.py run \
  --plan evidence/reacher-search-v1/protocol/plan.json \
  --expected-plan-sha256 8d5da5ee146c95532721fd1b6582bc37a091d5417e5d8b68f73e241379f2428b \
  --out runs/reacher-search-v1/execution
PYTHONPATH=src:scripts .venv-robotics/bin/python scripts/reacher_search_study.py audit \
  --plan evidence/reacher-search-v1/protocol/plan.json \
  --expected-plan-sha256 8d5da5ee146c95532721fd1b6582bc37a091d5417e5d8b68f73e241379f2428b \
  --execution runs/reacher-search-v1/execution \
  --out evidence/reacher-search-v1/audit
```

Output paths are exclusive. Do not rerun into an existing attempt, replace
cases, or extend the cap. The audit reconstructs proposals from saved scores
and replays native actions; it performs no new learned inference.
