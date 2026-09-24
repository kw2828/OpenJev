# Certificate-driven readout solver: synthetic engineering qualification

Version: `finite-gap-solver-qualification-v1`.

This is a new, bounded engineering study. The failed `finite-convex-readout-v1`
attempt remains closed: no retry, new tolerance, replacement parent, or use of
its reserved DEV namespace. This qualification uses no empirical arrays,
model checkpoints, learned recurrent states, environment calls or teacher calls.
Passing it does not admit an empirical successor or establish a learning result.

## Fixed question and methods

Test whether a conventional projected solver that stops on the required
direct-residual certificate satisfies that criterion across a predetermined
synthetic suite. Compare the unchanged SLSQP implementation with one new method,
`gap_projected`. No hyperparameter search or empirical fixture construction.

Both use the original four-action, eight-state readout, `C = .25 - P`, with each
column of P on the closed probability simplex. Loss is blind-cost MSE **plus**
observed-cost MSE, each divided by `4*N*H`. All absorbed zero-state rows remain
in both denominators. Targets are centered over actions. No regularization.

The original SLSQP solver retains maxiter 2,000, ftol 1e-12, maximum 10,000
objective calls, one final simplex projection with maximum repair 1e-10, and
its original success and direct-certificate conditions.

The new method uses projected FISTA with a deterministic monotonicity restart.
Set `L = 2 * max_i sum_j abs(Gram[i,j])` and use step `1/L`. This is a
conservative curvature bound for the symmetric quadratic. At a proposed
quadratic increase exceeding 1e-15, discard the extrapolation, reset momentum
and compute one ordinary projected-gradient step. If that step still increases
the quadratic beyond 1e-15, return a numerical failure. No line search,
backtracking, data-dependent budget change or returned-point repair.

Maximum new-method iterations: **20,000**. Check the direct residual certificate
at iteration zero, every ten iterations and the final iterate without duplicate
checks. Stop only with simplex violation at most 1e-12, signed Frank-Wolfe gap
between -1e-12 and **1e-8**, and direct objective at most initial objective plus
1e-12. For zero curvature, retain the initial point and require its certificate.
Any unsupported curvature, numerical inconsistency or exhausted budget stays a
failure. The numerical certificate is not an interval-arithmetic proof.

## Eighteen fixtures, generated only after registration

The first sixteen are the ordered Cartesian product:

1. Geometry: `basis`, `correlated95`, `correlated999`, `rank_deficient`.
2. Reference head: `interior`, `boundary`.
3. Target: `exact`, `misspecified`.

Each has N=16, H=2. Base directions are the eight basis vectors followed by a
copy rotated three positions. Horizon-zero mass is one; horizon-one mass cycles
through 0, .001 and .5. The observed route rotates cases by five. Correlated
directions mix basis with state-zero mass, using eps=.05 or .001. Rank-deficient
directions tie adjacent state columns. The source fixes the exact array order.

The interior reference head has .625 on the rotated preferred action and .125
on each other action. The boundary head is the corresponding vertex. Initial
P is uniform in every fixture. Exact targets are `X @ (.25-Pstar).T`.
Misspecified targets add centered deterministic row perturbations built from
rotations of `[.03,-.03,.01,-.01]`, with fixed case/horizon/route-dependent signs,
including perturbations on zero-state rows. These fixtures have no asserted
known optimum.

Finally, `zero_support` has zero states and nonzero centered targets with a
constant loss of 1/64; `small_mass` scales an exact basis/interior problem by
1e-8 in state mass. Both have analytically justified optimum references. No RNG,
training namespace, empirical arrays or model state determines any fixture.

Small-mass or constant-objective cases may satisfy the absolute gap at
iteration zero. Report that explicitly. Their inclusion demonstrates handling
of scale and zero information, not convergence from a difficult initial point.
Rank-deficient cases do not justify coefficient-recovery claims.

## Execution, independent checks and acceptance

Freeze this protocol, all source and test bytes, package versions, fixture
roster, resource bounds, output paths and method order before tests or fixture
generation. One exclusive registration and one original supervised engineering
attempt. No registered source edits during execution.

Run scoped lint and the three registered test files first. Then construct each
fixture once, save its raw float64 arrays and build its Gram statistics once.
Run both methods for each fixture, alternating method order by fixture index.
Both start from the same initial P, not the other method's solution. There are
36 planned solves. Method failures are saved and the remaining fixed roster
continues; a deadline, malformed output or independent arithmetic disagreement
stops the original attempt and retains all completed work.

Save each solver result before independent validation. A separate scalar loop
implementation operates directly on raw arrays and independently computes
initial/final loss, all gradient entries, signed gap and primal feasibility.
Verify claimed certificate values with absolute tolerance 1e-12 and relative
tolerance 1e-10. Admission uses the original absolute gap and feasibility
requirements, not those comparison tolerances. Known optimum points must
independently certify and match their analytic loss within 1e-12 (1/64 for
zero support, zero otherwise); each returned objective must be at most the known
optimum loss plus 1e-8 plus 1e-12 arithmetic slack where such a reference exists.

**Engineering PASS requires all 18 new-method solves complete, independently
pass the numerical certificate, not increase initial loss, and pass applicable
known-optimum checks.** Preserve all SLSQP results and failures. SLSQP need not
pass for the new method to qualify. A single failing new-method case prevents
qualification. Technical completion and the engineering gate are distinct.

Count every projected trial, restart, objective/gradient computation and direct
certificate pass. Report initial gap, iterations, each method's elapsed solve
time, separate shared Gram-build time and independent scalar-check time.
Fixture construction and initial/known-point scalar checks have separate
timers. Solver timings exclude prior model training, fixture construction and the
shared Gram build; those are not speedup or matched-compute claims. The outer
phase includes setup, tests, all fixtures/solves/checks and cleanup.

## Bounds and evidence

Native suspend-inclusive process cap: **600 seconds**, fixed before execution.
Worker peak RSS limit: 4 GiB, checked every 100 budget callbacks and at phase
boundaries. This is a worker check, not a whole-process-tree RSS measurement.
Output cap: 64 MiB and 256 paths, checked at file-writing boundaries. Numerical
libraries are restricted to one thread. No paid APIs, GPU or network data.

Retain registration, source pins, original supervisor launch/log/terminal,
worker receipt, lint/test logs, all eighteen fixtures, every completed solve,
the ordered independently checked journal and complete summary when reached.
The supervisor handles whole-process-group termination and cleanup. Source
identity is rechecked on normal completion; failure must not invent a missing
source-after attestation. No timeout, failed fixture or failed test is silently
rerun, replaced or removed.

If successful, a separately registered successor may revisit the readout
question with all nine original parents, original training labels, unchanged
heads, unchanged scientific criteria and a new development namespace. It must
disclose prior exposure to the original TRAIN data. Neither this qualification
nor another favorable average establishes a recurrent, connectome or ICLR
novelty result.
