# Sensitivity regularization: bounded prior-art check

**Proposed, unregistered and unrun. Source review dated 2026-09-25.** This note
reads four primary papers and the [existing residual proposal](fsm-residual-next.md).
It does not use live NL-LFR weights, measurements or outcomes. The original
[author-reference comparison](fsm-author-nllfr-protocol.md) must close before
any follow-up decision. Existing DEV is exposed; neither 300 mV nor official
TEST access is authorized here.

The broad idea is established: constrain or regularize recurrent sensitivity to
reduce error amplification. A linear model plus a constrained nonlinear residual
is also prior art. The useful question is narrower: **does penalizing sensitivity
added beyond our frozen linear model preserve useful predictive dynamics better
than penalizing total sensitivity?** This is a testable adaptation, not an
architecture-novelty claim.

## Four primary sources and their implications

| Primary paper | Verified mechanism and relevant limit |
| --- | --- |
| Miller and Hardt, *Stable Recurrent Models*, ICLR 2019 ([paper](https://arxiv.org/html/1805.10369), Definition 1, Sections 2.2 and 4.3) | Defines contraction of the state transition under a common input and enforces sufficient norm bounds during training. For a tanh RNN, recurrent spectral-norm projection is one route. It also examines data-dependent sensitivity and the potential cost of conservative global constraints. Regularizing recurrence to limit sensitivity is therefore not new; stronger contraction can sacrifice useful memory or prediction capacity. |
| Revay, Wang and Manchester, *Recurrent Equilibrium Networks: Flexible Dynamic Models with Guaranteed Stability and Robustness* ([paper](https://arxiv.org/html/2104.05942), Theorems 1–3 and Section VII) | Constructs models with contraction and optional incremental input/output gain bounds through certificates and direct parameterizations. Its empirically maximized sensitivity is explicitly a lower bound on the true gain, distinct from its certified upper bounds. A finite set of Jacobian probes cannot inherit REN's guarantees. No REN implementation is proposed as an extra family in this pilot. |
| Frank, Holicki, Scherer and Staab, *Regularised neural network-based nonlinear system identification with prior system knowledge*, 2026 ([publisher paper](https://doi.org/10.1080/00207179.2026.2679228), Sections 2, 5–7 and 9) | The closest hybrid precedent: augments known linear dynamics with an RNN residual and enforces dissipativity/gain constraints using matrix inequalities and barrier regularization. Its reported benefits do not apply uniformly across systems; its conclusion explicitly discusses limitations on MIMO generalization. Combining a linear approximation, neural residual and stability-oriented training cannot itself support a novelty claim. |
| Floren and Swevers, *A guided residual search for nonlinear state-space identification*, June 2026 revision ([paper](https://arxiv.org/html/2602.22964v2), Theorem 1 and Sections II-A/III-C) | Derives the recursive discrepancy bound `e[n] <= epsilon * sum(L^j)` under stated uniform Lipschitz/local-error assumptions and mitigates the recursion-free fitting mismatch through multiple shooting. Its residual-search penalty also measures residual influence through system matrices, rather than raw residual coordinates alone. Error amplification and system-aware residual penalties are established motivations. Our H128 free-running training already addresses a different training interface, so their result does not diagnose our present model as unstable. |

This limited search found relevant precedents, not evidence that the exact
penalty below is absent from the literature. A priority claim would need a
broader search and a clear technical contribution beyond this experiment.

## One concrete hypothesis

Keep the existing float64, order-32 frozen VARX plus width-24 tanh **feedback**
residual. Do not change initialization, data, recurrence, parameter count or the
H128 forecast-MSE objective. Let `q` contain the chronological 32 observed
three-channel outputs in the existing FIT-standardized units. Past and executed
future inputs are held fixed. Let `Phi_theta,h(q)` stack the first `h` predicted
outputs, and `Phi_0,h(q)` be the same rollout with the residual identically zero.
Both maps include the complete recursive propagation.

For `h in {8,32,128}` and two prospectively seeded unit Rademacher directions
`v in R^96`, define

```text
J_theta,h = d Phi_theta,h / d q
G_theta,h(v) = ||J_theta,h v||_2^2 / (3h)
G_0,h(v)     = ||J_0,h v||_2^2 / (3h)
R_excess = mean_(window,h,v) max(0, G_theta,h(v) - G_0,h(v))^2
L = L_H128_MSE + lambda * R_excess.
```

Directions use an independent, recorded random stream, shared across matched
recipes. They perturb only arrived output history, not applied inputs or future
targets. Differentiate through the full recurrent tangent calculation, including
its dependence on the residual weights; no detached Jacobians. Analytic tangent
propagation through the small tanh head can avoid materializing a full Jacobian.
Qualification should independently check the tangent against finite differences
and show that zero residual gives exactly the linear sensitivity and zero penalty.

The baseline-relative hinge allows reduced sensitivity without penalty and does
not directly penalize input-response gain. It does not ensure that useful input
responses remain unchanged. A directional average can miss harmful directions;
it is neither a spectral-norm estimate nor a worst-case certificate. The linear
baseline is a comparison point, not an assumed safe bound.

This output-history coordinate choice matters. Our lag-state update includes
unit-copy shift rows, making strict one-step contraction in its ordinary
Euclidean state norm structurally inappropriate for order greater than one.
Weighted metrics or multistep contraction are different possibilities, but are
not part of this proposal. H128 output sensitivity is also not a bound on the
amplification of errors injected at every intermediate step.

## What would make the experiment informative

Use two complementary sensitivity controls on the same windows, horizons,
directions and tangent implementation:

```text
R_total = mean G_theta,h(v)^2
R_envelope = mean_(window,h,v) max(0, G_theta,h(v) - max_v' G_0,h(v'))^2
```

Total sensitivity tests shrinkage without an allowance. The max-envelope uses
one shared threshold from the same two baseline directions at each window/horizon,
while the candidate uses each direction's own allowance. Both squared hinges
have zero penalty and gradient at the zero-head initialization. A mean-envelope
would not preserve this property when baseline direction gains differ: it would
already penalize the larger linear gain. The max-envelope is more permissive
than the directional allowance, however, so it does not match effective penalty
strength or feasible sets. A fixed-strength win alone cannot isolate directional
alignment.

Retain unregularized training, ordinary parameter L2, and residual-magnitude
penalties as simpler controls on the same model. These give the six arms in the
[prospective engineering-screen draft](fsm-sensitivity-experiment-draft.md).
Its one-coefficient screen is not a claim of superiority over tuned regularizers.
A later mechanism study would need equal declared strength-search budgets,
paired seeds/batches, initialization and optimizer updates; record derivative
time and total training cost as well. Keep the selected conventional references
unchanged. Neither document registers or launches a run.

A mechanism-specific interpretation needs a clean forecast-quality/cost benefit
and a predeclared context-perturbation diagnostic with unchanged future targets.
Use fresh diagnostic directions rather than training probes; freeze perturbation
units, magnitude and scoring before fitting. Report all records, channels and
seeds, including clean-error harm, nonfinite cases, penalty magnitudes and
hinge-active fractions. Lower probe gain alone is not success: a model can ignore
useful history or the excess penalty can remain inactive. Equal results from
simpler penalties weaken the distinction; lower sensitivity accompanied by worse
forecasts falsifies its intended predictive benefit. An effect confined to
corrupted contexts supports initialization-error robustness rather than better
clean dynamics.

Any eventual gain would support this local regularization hypothesis on the
registered task. It would not establish global stability, certified robustness,
untouched transfer, a new architecture, or proprietary Jev equivalence.
