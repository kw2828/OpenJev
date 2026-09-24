# Next prerequisite: reliable numerical termination

**Proposal only. The failed finite-convex-readout-v1 study remains closed.** Do not restart its six failed solves, relax its 1e-8 certificate, extend its budget, or publish selected successful parents as an evaluation result.

The run establishes a practical discrepancy: all nine solver success flags are true, while only three meet the registered direct-residual criterion. The implementation qualification included a correlated synthetic fixture, but that coverage did not establish reliability for this training design. Before another model comparison, test the solver's stopping behavior on a separately frozen synthetic engineering suite.

## Engineering question

Can a bounded convex solver stop directly on the same independently computed gap required for admission, rather than relying on a separate optimizer success flag?

Construct a fixed, model-independent suite before execution, covering correlated and nearly rank-deficient state features, small state masses, absorbed zero rows, boundary optima and interior optima. Keep analytically known solutions where possible. Include a separate scalar implementation for loss, gradient, feasibility and gap. Do not construct fixtures from saved TRAIN states or select synthetic cases by inspecting which parent failed.

Compare the existing SLSQP recipe with one predeclared method that uses the certificate as its stopping rule, such as projected gradient with a fixed step derived from the quadratic curvature and Euclidean simplex projection. Declare its objective, accuracy, iteration and elapsed-work limits in advance. The method is a numerical dependency, not a claimed learning contribution. Report all failed fixtures and total objective/gradient/projection work; avoid searching solver settings on the nine empirical cases.

## Reopening the scientific question

Only a separately registered successor can revisit the frozen-readout question. It must disclose that the original TRAIN data and parent models have already been inspected, preserve all nine parent/seed combinations and unchanged-head controls, retain the same accuracy requirement, and use a new declared development namespace. Original DEV namespace 427260924 was never generated, but leaving it reserved to the stopped study makes lineage clearer.

All training solves must pass before any successor development generation. If this numerical prerequisite fails, stop it as an engineering failure; do not infer a state-representation limitation. If a completed readout comparison later fails the all-seed decision criteria, change the scientific hypothesis rather than repeatedly optimizing this same head.

The broader recurrent/connectome goal remains open. A useful next architecture experiment still needs a mechanism, matched controls, scenario shift and a second environment. This numerical repair alone is not an ICLR contribution.
