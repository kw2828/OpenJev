# Capacity of fixed pose-expert outputs

The previous goal turn made progress: thirty new neural fits and an independent
audit established small benefits from excluded-parent selector training, but
the full continuation rule failed. Before adding another selector, measure
whether the existing expert outputs can support the required improvement.

An initial saved-output calculation already found that a perfect, hindsight
hard selector, choosing one expert separately for each endpoint and keeping
that choice throughout each forecast, can improve plain rotation by only 7.71%.
That cannot satisfy the unchanged 10% margin. The unrun hard-selector proposal
is stopped. This analysis formalizes that result and measures the larger class
of continuous mixtures. It is explicitly post-outcome and uses future targets.

## Frozen evidence and candidate classes

Use the sealed pose-crossfit-v1 fast and slow forecasts, paired targets and IDs:
two exposed archives, three fixed seeds, 160 windows per row, 25 steps each.
Authenticate its protocol, all execution payloads, completed corrected audit
and retained repair metadata. Keep all 33 canonical control configurations.
No model training, new model forecast, environment call or random draw occurs.

Compute three distinct hindsight quantities for each endpoint and window:

1. **Fixed hard choice:** choose the expert with lower average squared error
   over the entire 25-step forecast. Position and rotation may choose different
   experts. This is the best possible fixed hard routing for these outputs.
2. **Per-step hard choice:** choose independently at every future step. This
   relaxes fixed hard routing, but is not a bound for continuous mixtures.
3. **Continuous mixture:** choose one position coefficient and one rotation
   coefficient in [0,1], each fixed throughout the 25-step forecast. Optimize
   position with exact clipped least squares. For rotation use the unchanged
   pose-constant-v1 interval search on float64 SVD-projected SO(3) inputs.

Run all 960 per-window constant searches with tolerance 1e-7 rad² and a cap of
4,095 objective evaluations per search. Retain every interval/evaluation trace,
identity, coefficient, bound, timing and certification status. A capped search
retains its valid recorded bounds and remains visibly uncertified. Do not
retry, change budgets, substitute cases or select an advantageous subset.
Require the externally recorded protocol digest at both execution boundaries.
On failure, preserve any returned searches from the incomplete row separately;
do not resume that run. The evaluation cap applies per search, with no declared
overall wall-time cap.

## Comparison semantics

Pool squared errors over all 480 seed-window pairs per panel before taking the
square root. Compare against each of the 33 controls' reported physical MSE.
The unchanged 10% RMSE margin is an MSE threshold of 0.81 times the control MSE.
Equality meets this necessary margin; exclusion requires a strict inequality.

For fixed hard routing, error above a threshold means even perfect future
knowledge cannot meet that necessary condition for the fixed outputs. For
continuous position, the least-squares optimum gives the analogous exact
numerical result. Rotation lower and upper bounds apply only to the projected
mathematical interpolation. Their comparisons to reported control thresholds
are explicitly labeled projected comparisons. They do not provide a uniform
bound for the original float32 production interpolation or its metric.

An upper bound below a reference threshold demonstrates hindsight capacity only.
A lower/upper interval straddling that reference remains unresolved at the
declared search budget. Even favorable bounds do not establish an implementable
selector, the paired/parent/leave-one-out requirements, latency, a full gate
pass, generalization or architectural novelty. No success gate is assigned to
this oracle diagnosis. Per-step hard results must not be described as bounds
on continuous mixtures, which can outperform both endpoints.

## Evidence and independent verification

Freeze thirty source files and the parent evidence before the continuous
capacity calculation. Save twenty execution files: start/completion records,
six compressed full trace files, six numeric bound arrays and six row records.
An independent saved-output audit verifies all identities and all 960 complete
search certificates without reoptimizing, using the pinned earlier geometric
audit and its reviewed signed-scalar check. It independently reconstructs both
hard classes and all pooled comparisons, and retains certification failures.

Publish numerical bounds, coefficients, traces, code, figures and receipts.
Original target and forecast arrays remain in their previously sealed local
archives under unresolved upstream licensing. The result bounds these fixed
expert outputs and horizon-constant coefficients. It does not exclude improved
experts, time-varying continuous mixtures, changed recurrent dynamics or other
representations. The purpose is to decide whether further selector work has
enough numerical room, not to lower the research objective to an oracle score.
