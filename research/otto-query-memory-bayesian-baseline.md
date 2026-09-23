# A stronger Bayesian memory control

**Proposal only.** The current query-memory experiment remains frozen. This
note specifies a known baseline for a separate study; it reports no new run,
performance gain, calibrated probability or architectural novelty.

The question is whether accumulated evidence should change how strongly memory
responds to the next labeled observation. Compare the current normalized delta
update with recursive least squares (RLS) around the same frozen recurrent
predictor. This would isolate the online estimator before adding a query gate,
reinforcement learning or a new backbone.

## Three relevant precedents

| Primary source | What it contributes to this comparison |
| --- | --- |
| [ALPaCA, sections 3-4 and equations 13-14](https://arxiv.org/pdf/1807.08912) | Offline learning of features and a Bayesian linear prior, followed by recursive Bayesian regression with fixed online features. Its robotics experiments make it a relevant baseline, not a novelty claim for our proposal. |
| [Longhorn, equations 6-9](https://arxiv.org/html/2407.14207v2) | Associative memory derived from an implicit online regression update. Its implemented recurrence uses a diagonal approximation to the exact transition. It does not maintain the posterior covariance proposed below. |
| [Synaptic plasticity as Bayesian inference, Figure 4 and Methods equations 57-60](https://www.gatsby.ucl.ac.uk/~pel/papers/bayesian_plasticity_2021.pdf) | Uncertainty-dependent learning of output weights with fixed recurrent and feedback weights. This recurrent experiment assumes no target-weight drift; the broader paper also studies drifting synapses. It supplies biological motivation and close prior art, not a connectome implementation. |

## Proposed control in our coordinates

Let $C=I_4-\mathbf{1}\mathbf{1}^{\top}/4$, the centered teacher residual be
$r=C((Q-b)/64)$, and the history cue be $z\in\mathbb{R}^8$. Here $b$ is the
slow predictor's full shadow prior. Keep the projection fixed online.

Represent the memory mean by $W\in\mathbb{R}^{4\times8}$ and a shared weight
covariance by $P\in\mathbb{R}^{8\times8}$. Assume independent Gaussian noise
with variance $\sigma^2>0$ in each coordinate of an orthonormal three-dimensional
contrast basis. This is equivalent to output noise covariance $\sigma^2 C$.
With $W_0=0$ and $P_0\succ0$, the standard Bayesian linear update is:

$$
k=\frac{Pz}{\sigma^2+z^\top Pz},\qquad
W^+=W+(r-Wz)k^\top,\qquad
P^+=P-\frac{Pzz^\top P}{\sigma^2+z^\top Pz}.
$$

All right-hand quantities use the pre-update state. Centered residuals preserve
centered columns of $W$ in exact arithmetic. This is the covariance form of
recursive Bayesian regression; ALPaCA factors observation noise out of its
inverse-precision matrix instead. No numerical prior or noise setting is chosen
here.

Read before writing. Update both $W$ and $P$ only on actual labeled queries,
with the same first-query exclusion as the delta control. Seeing an unlabeled
cue alone must not shrink posterior covariance. Epistemic output covariance is
$(z^\top Pz)C$; predictive covariance adds $\sigma^2 C$. Neither is a calibrated
probability that an action is correct. Correlated cues are compatible with
conditional regression, but correlated residual noise, changing coefficients
and misspecification can invalidate the uncertainty interpretation.

Our algebra also shows why renaming the existing delta update would add little.
Its coefficient $\eta/(\epsilon+\|z\|^2)$ equals Longhorn's exact proximal
coefficient $\beta/(1+\beta\|z\|^2)$ when
$\beta=\eta/(\epsilon+(1-\eta)\|z\|^2)$ for $0<\eta\le1$ and $\epsilon>0$.
That identity concerns the exact proximal update, not Longhorn's diagonal
implementation. RLS adds persistent directional precision.

## What a separate experiment should resolve

Hold the backbone, cue projection, centered targets, training exposure and
teacher-query schedule constant. Retain ordinary recurrent, last-error and
normalized-delta controls. Compare full RLS with a diagonal-covariance
approximation to determine whether cross-direction information earns its cost.

In a dense float32 implementation, full $P$ adds 64 scalars (256 bytes). With
the existing 32-scalar mean and eight-scalar trace, memory grows from 40 to 104
scalars, excluding the backbone and work buffers. Covariance updates and
uncertainty reads cost $O(d^2)$; the existing mean update costs $O(4d)$.
Identical teacher access does not make these methods equal in total compute.
Measure that cost rather than claiming a speed advantage from operation counts.

The hypothesis is better later predictions when cues revisit similar directions.
Also test fresh scenario shifts: without forgetting, reduced covariance may
make adaptation too slow. Freeze any prior/noise selection, development budget
and acceptance rule before those evaluations. Standard forgetting or process
noise would be additional established controls, not novel mechanisms.

If RLS explains a gain, report Bayesian residual adaptation. A later experiment
could test whether its uncertainty helps decide when to pay for an expert, with
matched total compute and autonomous outcomes. The present proposal establishes
neither RLCD nor a world model, conformal coverage, biological superiority or an
ICLR contribution.
