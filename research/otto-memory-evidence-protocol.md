# Saved-history diagnostic: what older odor evidence changes decisions?

21 September 2026. This is a **post hoc mechanistic diagnostic**, specified after
the completed [53x53 comparison](otto-large-memory-results.md). That comparison
reduced average search time 41.13% but failed its consistency condition. This
analysis cannot reverse that failure, admit a learned pilot, establish a better
controller, or serve as independent confirmation.

## Question and data boundary

Identify whether older positive detections, older zero detections, or their
combination change decisions relative to the last 32 observations. Use only the
saved, authenticated 96 primary space-aware full-history trajectories and their
96 recent-32 counterparts. Do not sample an environment, train a model, call an
external model, change evaluation seeds or extend the completed cohort.

Authenticate the original plan, completed worker receipt, completed supervisor
terminal and agreeing independent audit with externally supplied hashes. Bind
the required original payloads through those receipts. Freeze this protocol,
the diagnostic implementation, its synthetic tests, the existing independent
analytic auditor and suspend-inclusive clock in a new plan before execution.
The diagnostic may reuse the auditor's analytic likelihood, prior and one-step
action scoring functions. It must not import the original simulator, runner,
actor or policy classes.

## Four fixed evidence constructions

At a decision after `t` completed nonterminal moves, all constructions receive
the same initial-hit-conditioned prior and permanent mask of cells visited
without finding the source. The initial hit is always included once and never
belongs to the ablated old observations. Define recent observations as the last
`min(t,32)` completed readings. Earlier completed readings form the old set.

1. **Recent 32:** apply only the recent likelihoods.
2. **Recent 32 plus old positives:** also retain every old reading in categories
   1, 2 or 3. Category 3 includes the saturated tail of at least three detections.
3. **Recent 32 plus old zeros:** also retain every old reading in category 0.
4. **Full:** retain both old sets and the recent readings.

Reconstruct each belief from the shared initial prior, exclude all known
non-source visited cells, apply retained likelihoods in chronological order,
and normalize without smoothing or probability floors. Validate finite,
nonnegative unit mass and zero mass on visited cells. Reject invalid histories.
The public reconstruction function receives only the initial public packet and
completed public observations, never a source coordinate, random seed, draw log,
saved simulator posterior or future observation.

This is a fixed 2x2 decomposition of old evidence, not a sweep of windows or
retention hyperparameters. Indefinitely retaining one evidence category is not
necessarily compact memory and is not itself a proposed novel architecture.

## Two analysis populations

**First divergence on shared history.** For every full/recent-32 case pair, locate
the first differing issued action, if any. All preceding actions and public
packets must be identical. At that common prefix, compute all four constructions.
Report all 96 cases, including cases with no divergence. Do not compare beliefs
from two different subsequent trajectories as though they shared observations.

**All full-history decision prefixes.** Reconstruct the full-history belief and
its four action scores at every pre-action prefix of each primary trajectory.
Require exact selected-action agreement and maximum score error at most `1e-8`
against the saved full-history calculation. Analyze the four constructions at
every decision with more than 32 completed observations, without selecting by
the realized outcome or magnitude of disagreement. Keep denominators for all
decisions, eligible prefixes, eligible cases and initial-hit strata.

Transition row `k` stores the action and scores chosen before move `k`, but its
public packet is the observation after that move. A decision after `t` completed
moves therefore uses row `t+1` scores and only observations through row `t`.
The terminal found sentinel is never an odor observation or decision input.
The same original first-action-within-`1e-10` tie rule applies.

The second population is a **teacher-forced diagnostic**: all constructions see
the full-history controller's saved path. Their proposed actions are not
executed. Agreement with full history is not accuracy or improved search, and
full history is not an optimal policy reference.

## Fixed outputs and interpretation

For each eligible prefix retain:

- Selected action and all four action scores for every construction.
- Total variation distance of each belief from the full-history belief.
- Difference between the full-belief heuristic score of that construction's
  chosen action and the minimum full-belief action score. This is a dimensionless
  surrogate, not realized search-time regret or a learned value estimate.
- Counts of old positive and old zero observations and the elapsed moves since
  the last positive reading, including the initial hit as time zero.

Publish prefix rows, per-case aggregates and all case identities. Report
unweighted eligible-case means, stratum-weighted eligible-case means and
stratum-weighted eligible-prefix means separately. Case weight is the original
initial-hit mixture weight divided by 32. Conditional means renormalize weights
over their explicitly stated eligible population; cases without eligible
prefixes contribute to coverage counts but not conditional metric means.
For first-divergence summaries report both counts and mixture-weighted rates
conditional on divergent cases. Report ties and disagreements without removing
unfavorable cases. No pass/fail performance threshold is added.

A separate simple saved-outcome decomposition may describe paired time gains,
their fixed-block and initial-hit contributions, largest-contributor sensitivity,
and leave-one-case/block-out means. Source distance, if used there, is explicitly
evaluator-only and never enters the evidence reconstructions. These descriptions
are post hoc and do not constitute uncertainty intervals or causal explanations
for entire episode returns.

## Execution bounds and continuation

One saved-data invocation, in an exclusive output directory, with a 180-second
suspend-inclusive wall cap, 2 GiB peak RSS cap and 64 MiB output cap. Preserve a
failure receipt on any violated bound or validation failure. Do not overwrite
outputs or silently retry a failed invocation. Record source/input hashes,
actual processed counts, numerical parity, elapsed time and terminal status.
An implementation failure may be corrected only as a separately recorded
attempt, retaining the original failure; it does not authorize new experiments.

If one category explains the first divergences, that motivates a narrower
mechanistic hypothesis to test with new, prospectively specified controls. It
does not prove that retaining that category improves autonomous search. If both
are needed, avoid a selective-retention narrative. Any later learning or control
study requires its own justification and protocol; the original 9/10 failure
remains the authoritative continuation decision.
