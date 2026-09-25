# Prior art and interpretation of the FSM residual comparison

**Design reference, not a performance result or an architecture novelty claim.**
The experiment protocol governs execution. This note uses primary literature
and source contracts; it does not introduce measurements or authorize test access.

## What is already established

- [NL-LFR initialization from a best linear approximation](https://arxiv.org/html/2004.05040)
  separates linear dynamics from a static neural nonlinearity in feedback.
  Its initialization suppresses the nonlinear contribution so the initial model
  reproduces the linear approximation. Linear initialization plus a learned
  residual is therefore established, not a new recurrent architecture.
- [dynoNet](https://arxiv.org/abs/2006.02250), with
  [official code](https://github.com/forgi86/dynonet), combines differentiable
  rational dynamical filters with static nonlinearities. It is a relevant
  compact nonlinear baseline, not evidence that a generic GRU is the strongest
  available identification method.
- [SUBNET](https://arxiv.org/html/2210.14816) uses a history encoder and
  multistep subsequence losses. Its
  [linear-initialization extension](https://arxiv.org/abs/2304.02119) initializes
  both state dynamics and the encoder from a linear approximation. Neither
  the use of observed history nor linear initialization alone establishes novelty.
- [Floren and Swevers' guided residual search](https://arxiv.org/html/2602.22964)
  explicitly analyzes the shift between recursion-free residual fitting and
  recursive simulation. Small local errors can compound under feedback;
  improved one-step fitting need not improve a long rollout. Their method uses
  guided initialization followed by multiple-shooting refinement. Our direct
  free-running training is a simpler comparison, not a reproduction of it.

## What this factorial can establish

The four arms cross **affine / width-24 tanh residuals** with
**output-only / feedback placement**, using the same frozen order-32 VARX
backbone, 195 lag/current-input features, float64 arithmetic and zero output
initialization. Output-only evolves the uncorrected linear trajectory;
feedback inserts corrected outputs into subsequent lag states. Both use the
last 32 observations and past inputs of the supplied C100 context. Future
input at a step may affect that step's prediction; future outputs are unavailable.

All arms use full H128 free-running standardized-output MSE, no auxiliary loss,
no weight decay, clipping at 1, 2,048 updates and batch 16. The predefined rates
are **1e-4, 3e-4 and 1e-3**, with three seeds and all **36 fits** retained.
One rate per architecture is selected by its three-seed mean on exposed DEV.
This is a matched search-budget comparison, not untouched confirmation.

The affine control matters because the frozen VARX was fitted using one-step
ridge loss. A residual can improve multistep prediction by changing the effective
linear model, without learning useful nonlinearity. Parameters match within
each placement pair: 588 for affine and 4,779 for tanh. Across those types,
capacity differs, so a tanh advantage does not isolate nonlinearity from capacity.
Identical initial forecasts also do not imply identical training gradients:
feedback changes the differentiation path through the rollout.

Residuals are added in FIT-standardized output units before denormalization.
Report physical errors separately and charge complete-request latency and all
retained coefficients, weights, normalization and state. A frozen backbone is
not a stability guarantee for its nonlinear feedback extension. Quality and
cost outcomes should remain separately visible even when the overall rule fails.

The affine-feedback residual can be algebraically folded into effective VARX
coefficients. The current 36-fit / 37-evaluation design retains its unfused
PyTorch form for the placement comparison, so its raw latency overstates the
minimum deployable linear-control cost. The accuracy comparison remains valid;
the latency gate compares only the matched tanh pair. No strong neural runtime
or accuracy-cost frontier advantage should be claimed until a folded native
linear reference has been qualified and measured.

## Only a speculative next mechanism

If correction propagation first shows useful predictive benefit, a later study
could test whether penalizing **added finite-horizon output sensitivity** improves
robustness more than ordinary weight decay or a matched residual-magnitude penalty.
Sensitivity must use declared standardized units and the complete recurrent map;
finite-horizon diagnostics alone would not certify stability. Related
[stability-constrained hybrid residual models](https://doi.org/10.1080/00207179.2026.2679228)
already exist, so any claimed contribution needs a precise distinction and controls.
Biological error gating could motivate a hypothesis, but supplies neither evidence
nor novelty. No such gate, regularizer or follow-up experiment is established here.
