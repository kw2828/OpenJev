# Information available for fixed-horizon teacher decisions

Prospective diagnostic `otto-cost-information-v1`. Freeze source, inputs, seeds,
qualification and this protocol before empirical execution. The completed
paired-action study remains DEV_FAIL 6/18 under both of its gates. No fitting,
checkpoint selection, old TRAIN/DEV array reuse, TEST, replacement cases or
scientific retry is admitted here. Technical and budget failures are terminal.

## Question

Can a decision rule using the full observed public history, but no future odors,
meaningfully improve on the frozen recurrent cost head? How much of the absolute
teacher-imitation gap reflects the teacher seeing those future observations?

The latter term is shared by two policies on paired histories and cancels in
their difference. It cannot itself explain the preceding candidate's shifted
regression. The proposed diagnostic tests headroom and target noise, not that
causal explanation. The public-history reference also has richer information
than the student's 31-feature prefix. Its floor is not automatically the floor
for the student's compressed inputs.

## Frozen policies and fresh cases

Use all twelve final checkpoints from `otto-action-effect-v1`, four families and
fit seeds 330000001..330000003. Their parent closure is pinned to
`63973e02c90224a7224f4626af3ddc17509d5ee18e6f5aa0625dd0b5863f8d2d`.
Authenticate original fit receipts and checkpoint bytes. No parent data array
is admitted. Every policy receives only the same nine 31-feature prefix rows,
prefix length and eight proposed actions. No teacher belief, sampled source,
future odor or new conditional-cost label enters neural inference.

Attempt 32 fresh native starts per setting: lambda3 seeds331000001..331000032
and lambda4 seeds332000001..332000032. Initial hit cycles1,2,3. Observe eight
analytic-policy transitions exactly as in the parent collector. Exclude and
record prefix termination without replacement. Require at least16 retained
prefixes per setting. There is no native continuation after that prefix.
Precommit eight actions using PCG64(SeedSequence([environment_seed,911])).

In complete roster order, selection streams use seeds333000001..333000064;
evaluation streams use334000001..334000064. Bootstrap seed335000001. The scoped
193-seed metadata review must precede execution. Full global independence from
unknown external runs is not claimed.

## Two public beliefs and one explicit numerical reference law

Keep the unchanged legacy public filter for teacher inputs, including its
normalization floor and any subnormalized beliefs. Independently update the
strict shadow posterior under the authenticated source-draw/sensor law. Never
initialize the legacy filter from the strict posterior or substitute a teacher
call on an averaged belief for averaged teacher calls.

For the diagnostic, apply the existing 53-bit CDF conversion once to the strict
prefix posterior. Call the resulting distribution root_grid. Save root_strict,
root_grid and their total variation distance; require distance at most1e-10.
Both exact enumeration and Monte Carlo use root_grid. This is an explicitly
quantized numerical reference, not a claim of exact equality to the unrounded
conditional law. Do not infer a universal expected-cost error bound from its
small probability perturbation: teacher costs have no registered global cap.

The sensor law uses the same authenticated effective 53-bit category masses as
the parent study. Each source/odor draw uses saved integer uniforms obtained
from PCG64 random_raw shifted right11 bits. Map against cumulative integer bin
counts totaling2^53. Do not apply a second floating CDF normalization. Allocate
all128x9 draws in each stream before any terminal shortcut. Sampled source cells
are hypothetical reference draws; the actual native hidden source cannot enter
this procedure or policy inference.

## Exact horizon4 and sampled horizon8

For horizon4, enumerate all four-category odor histories from the same prefix
and fixed path. Carry unnormalized source mass, remove newly visited source
cells into absorbing-found mass, and multiply by each odor likelihood. Revisited
cells cannot double-count found probability. Do not prune tiny positive branches.
Only positive surviving leaves receive the original teacher's four costs, using
that branch's unmodified legacy filter. At most256 teacher calls per prefix.
Persist canonical base4 history codes, masses and all four raw costs. Found
leaves have zero mass/cost placeholders and no teacher call.

Generate128 hypothetical histories per independent stream out to horizon8.
Found is absorbing and skips later teacher calls, but consumes the already
allocated draws. At horizon4, look up costs from the exact leaf table; this must
match the first four sampled odors exactly. At horizon8, evaluate the teacher
only on surviving histories. At most256 further calls per prefix. No fabricated
counterfactual outcomes are represented as actual native trajectories.

Report exact horizon4 expectations alongside each stream's Monte Carlo errors
and estimated standard errors. Per-draw lookup equality is a hard audit check;
closeness of noisy means is descriptive, not an arbitrary acceptance threshold.

## Estimands and separate selection/evaluation

At a fixed horizon h, public prefix H, committed path u, and common geometry-
defined legal set, let Q_a(y) be the teacher's cost after surviving history y.
Found histories contribute zero to decision regret. Define

```text
d_a = sum_surviving_y p(y | H,u) * (Q_a(y) - min_b Q_b(y))
D = d_student_action
F = min_a d_a
A = D - F
```

D=F+A is the fixed-horizon unconditional decomposition. Report exact survival
mass S_h from the source law and committed path separately. At horizon4,
conditional values divide exact unconditional quantities by exact S_4. At
horizon8, descriptive plug-in conditional quantities divide empirical sums by
the evaluation stream's empirical survival fraction, not by exact S_8; they are
undefined if that stream has no survivors. Average
unconditional quantities equally over all retained originating prefixes,
including zero-support prefixes as zero. Never replace this by conditional
averages times average mass. The old study's random mean over surviving rows
across horizons is a different estimand and remains unchanged.

Horizon4 F is exactly enumerated under the declared grid law, subject to recorded
float arithmetic. Horizon8 plug-in decompositions on evaluation draws are
explicitly descriptive empirical quantities. Their adaptive minimum is not an
unbiased estimate or proof of the true information floor.

For the primary horizon8 comparison, choose the lowest-index minimum mean-cost
action using only the128 selection draws. Found samples have zero common cost,
so they cannot favor an action. If no selection samples survive, choose the
lowest legal index and retain that case. Evaluate this fixed selected action
only on the separate128 evaluation draws, shared with all frozen models.
This gives an achievable reference decision, not the exact optimal action.

The primary control averages the regrets of the three paired_recurrent fits as
three separate decisions. It does not average their logits/costs into a new
policy. Other frozen families and all fit-level quantities remain descriptive.

## Prospective headroom criterion

For each prefix/draw, let C be mean regret over the three frozen primary-control
actions and R be regret of the selection-stream reference action. Let gain=C-R.
Found draws contribute zero. Use equal prefix weights, then equal evaluation-
draw weights, including zeros. Selection remains fixed for inference.

Use2000 hierarchical bootstrap replicates per setting. PCG64 seeded with
SeedSequence([335000001, setting_index]) first produces all prefix-resampling
indices [2000,N], then all within-prefix evaluation indices [2000,N,128]. Share
those resampled evaluation histories across all compared actions. Repeated
prefixes receive independent inner resamples. Use linear quantiles2.5%/97.5%.
These are approximate percentile intervals, conditional on the realized
selection streams and fixed trained models. They are not rigorous population
bounds, do not include selection-algorithm variability, and fit seeds are not
independent datasets.

Require a positive lower endpoint for gain-0.10*C in lambda4 and for gain in
lambda3, positive mean C, at least16 prefixes in each setting, and at least32
surviving samples in each stream for every positive-S_h prefix. Insufficient
support or unresolved intervals produce HEADROOM_NOT_RESOLVED without retries.
A resolved outcome supports separately registering a conditional-label training
pilot. It does not itself authorize a new run or establish model improvement,
autonomous performance, architectural novelty or an ICLR-ready result.

## Resources and verification

Collection, prediction/analysis, and independent audit each use an original
supervised process with1,800second suspend-inclusive cap,4GiB RSS,512MiB output,
and single numerical threads. Native maximums include64 resets,512 prefix steps,
32,768 teacher calls and corresponding actual TensorFlow calls, plus fixed
filter/tree/draw caps in collector source. No extra teacher calls for MC horizon4.
Keep full costs and source lineage for the existing12models visible; no new
training speed claim. Collection capacity is a planning estimate based on the
previous measured teacher time plus fabricated full-size overhead, not a new
teacher benchmark or a guaranteed upper runtime bound.

Before registration, qualify pure branch arithmetic, absorbing/revisited states,
integer-bin sampling, conditional/unconditional decompositions, selection/eval
separation, support failures, bootstrap work and resource estimates on fabricated
inputs. Native preflight authenticates metadata without importing frameworks.

Audit saved arrays and source metadata independently. Reconstruct strict/legacy
prefixes, reference laws, exact branch masses and MC histories/leaf lookups;
save the unchanged native legacy kernels alongside the sensor-law tables so
the audit can reproduce the legacy filter without importing the native engine.
reconcile reported decision choices, metrics, bootstrap and criterion. No model,
teacher or native step runs in the audit. Teacher costs are authenticated saved
outputs, not independently re-evaluated. Verify original process closures and
separate audit closure before publication. Preserve all failed engineering or
scientific attempts and forbid outcome-driven replacement or budget extension.
