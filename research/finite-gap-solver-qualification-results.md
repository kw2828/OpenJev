# The certificate-driven solver passes the synthetic prerequisite

**18/18 fixtures qualify with the projected solver, compared with 10/18 for
the unchanged SLSQP recipe.** All 36 results were preserved and checked with
separate scalar arithmetic. This is a numerical engineering result, not better
gameplay, a recurrent-model improvement or a new algorithm.

The new solver uses fixed-step accelerated projected gradient with a
monotonicity restart. It stops on the same direct-residual accuracy criterion
that halted the previous empirical diagnostic. The threshold stays at 1e-8;
the model, objective and closed-simplex readout remain unchanged.

The deterministic suite covers basis, correlated and rank-deficient state
features, interior and boundary heads, exact and misspecified targets, zero
support, and tiny state masses. Each of the 16 nontrivial new-method solves
requires updates. The zero-support and tiny-mass cases pass at iteration zero,
so they are handling checks, not evidence of convergence or coefficient
recovery. The hardest new-method solve takes 9,630 iterations, below the fixed
20,000-iteration cap. The more demanding accuracy can require more computation.

For every result, a separate raw-array scalar implementation checks objective,
every gradient entry, feasibility and the signed Frank-Wolfe gap. Known optimum
points also match their analytic objective. No reference solver output is used
as the other solver's starting point. Alternating method order does not turn
these measurements into a matched-compute speed comparison.

The original phase completes in **3.042 seconds**, including lint, **73 passing
tests**, fixture generation, both solvers, independent checks and cleanup. All
12 source pins and 58 output payloads match their original receipt. There are
zero empirical array loads, model calls, checkpoint loads or DEV generations.
No settings, data or budgets changed after registration.

[Frozen protocol](finite-gap-solver-qualification-protocol.md) ·
[Separate empirical successor protocol](finite-gap-readout-study-protocol.md)

Registration SHA256:
`3513fd540b1a01af6568a4268a8ea7e0a93680ad39ef35ddb206d998cbf255eb`.

This justifies testing the qualified solver in a new registered diagnostic.
It does not prove that the readout explains the earlier recurrent failures.
The [stopped predecessor](finite-convex-readout-results.md), its six failed
certificates and its unused development namespace remain closed.
