# Causal state initialization for the author NL-LFR reference

**Proposed, unregistered numerical contract.** This is a source-derived adapter
design, not a measured result or a new architecture. No measurement arrays,
author checkpoints, model calls or fits were used to write it. Freeze the
constants and qualify the implementation on fabricated systems before applying
it to a fitted reference. The iteration bound below limits computation, not
the magnitude or stability of the latent state.

## Exact model and time convention

The pinned author [NL-LFR simulator](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_model_structures.py#L422-L450)
uses column states and the explicit equations

```text
z_t = C_z x_t + D_zu u_t
w_t = f(z_t)
y_t = C_y x_t + D_yu u_t + D_yw w_t
x_(t+1) = A x_t + B_u u_t + B_w w_t.
```

Output and next state both use the same pre-update state. There is no
instantaneous feedback from w into z and no algebraic fixed-point solve. For
the proposed two-hidden-layer ReLU network,
`f(z)=W3 relu(W2 relu(W1 z+b1)+b2)+b3`; the final layer is linear.
The author's [network wrapper](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/static/_nonlin_funcs.py#L118-L229)
uses Equinox's default identity final activation. Use derivative masks
`a > 0`, including derivative zero at an exactly zero hidden preactivation,
matching [JAX ReLU](https://docs.jax.dev/en/latest/_autosummary/jax.nn.relu.html)
and [Equinox MLP](https://docs.kidger.site/equinox/api/nn/mlp/). Validate the
exported activation, layer shapes and biases; do not silently approximate a
different static nonlinear function with this adapter.

A C100 request starting at s contains `y_s,...,y_(s+99)` and only
`u_(s+1),...,u_(s+99)`. Estimate `x_(s+1)` from those **99 paired samples**.
Exclude `y_s`: its matching input is unavailable. Never fill in `u_s`, wrap a
period, append a periodic tail, or expose future inputs/outputs to conditioning.
Normalize the context using the final model's own frozen FIT statistics.
Keep the common benchmark scoring normalizer separate.

## Seed in the final model's coordinates

Use the final NL-LFR's `A,B_u,C_y,D_yu` to construct the linear observability
system from the [qualified linear derivation](fsm-author-causal-contract.md):
`O_j=C_y A^j`, `f_0=0`, `f_(j+1)=A f_j+B_u v_j`,
`b_j=y_(s+1+j)-C_y f_j-D_yu v_j`. Solve
`x_seed=lstsq(O,b,rcond=1e-12)` in float64, with minimum Euclidean norm.
Here `v_j=u_(s+1+j)` and `j=0,...,98`.

This is a seed for **x at the second observed output**, not the state returned
by the existing linear `condition`, which has already advanced through the
context. Do not invert 99 transitions to recover that earlier state.
The final linear submodel is allowed to be unstable or poorly observable;
report its numerical rank and singular values rather than replacing it.

The author [optimizer](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_nonlin_lfr.py#L536-L562)
retains `_bla` and uses its periodic steady-state trajectory during training.
It jointly changes the final dynamic matrices and network, then restores that
original `_bla`. No explicit state-coordinate normalization is performed by
this nonlinear optimizer, but the original BLA state is not an inferred state
of the final nonlinear law. This request adapter must not silently use `_bla`
as its final-model state estimator or use the author's periodic training warmup.

## Context objective and analytic Jacobian

For a trial initial state x, simulate all 99 known inputs and stack the three
output residuals at each time, in time-then-channel order:
`r(x)=stack(y_hat_(s+1+j)(x)-y_(s+1+j))`,
`F(x)=||r(x)||²/(2m)`, where `m=99*3=297`.
Only the 28 initial-state coordinates vary; model weights and normalization
stay fixed. There is no penalty pulling the state back to the seed.

Let `S_0=I` and let `G_j=df/dz` on the currently selected ReLU pieces. Then

```text
G_j = W3 diag(a2_j > 0) W2 diag(a1_j > 0) W1
H_j = C_y + D_yw G_j C_z
F_j = A   + B_w  G_j C_z
J_j = H_j S_j
S_(j+1) = F_j S_j.
```

Stack `J_j` to obtain the 297-by-28 context Jacobian. The input is fixed while
differentiating the initial state; its effect still enters every preactivation
and propagated state. ReLU kink masks define a derivative convention, not a
smoothness claim. Do not compute `J_j` after advancing `S_j`.

## One bounded solver policy to freeze

Use at most **16 outer directions**, each with at most **8 trials**. Define
`rbar=r/sqrt(m)`, `Jbar=J/sqrt(m)`. At each outer iteration let
`c_i=||Jbar[:,i]||`, `cmax=max(c_i)` and
`d_i=max(c_i,1e-8*cmax)`. If `cmax=0`, stop with `ZERO_JACOBIAN`; do not invent
a scale or perturb the state. Otherwise set `K=Jbar diag(1/d)`.

If `||K.T rbar||_inf <= 1e-8*(1+||rbar||)`, stop with `GRADIENT_TOL`.
Otherwise solve the augmented least-squares system

```text
[ K             ] p ~= [ -rbar ]
[ sqrt(1e-3) I  ]      [    0   ]
delta = p / d
```

using `lstsq(..., rcond=1e-12)`. This is damping of the step, not a state prior;
avoid squaring the Jacobian's condition number through normal equations.
Try step sizes `alpha=1,1/2,...,1/128` in that order. Accept the first finite
trajectory satisfying `F(x+alpha*delta) < F(x)` and
`F(x+alpha*delta) <= F(x)+1e-4*alpha*(Jbar.T rbar).T delta`.
A nonfinite trial is explicitly rejected and counts against the eight trials.
Record all trial outcomes. A nonnegative directional derivative or exhaustion
of finite descent trials stops with `STALLED`; 16 exhausted directions gives
`ITERATION_CAP`. There are no random restarts, alternative seed selection,
adaptive jitter, parameter/state clipping, solver replacement or timeout retry.

Return the last accepted finite iterate and its finite 99-step terminal state,
including the seed if no step was accepted, with the true stop reason. This is
the prescribed bounded solve, not a switch to a separate BLA forecast. A finite
stalled/capped estimate may be scored but must remain visible as such; none of
these stop reasons certifies global or local optimality. Nonfinite seed/current
trajectory/Jacobian/linear-solve results, failed factorization, or a nonfinite
final forecast is an explicit failed request, never a hidden zero-state fallback.
Structural errors remain implementation failures.

The returned state is `x_(s+100)`, after all 99 nonlinear context transitions.
The first forecast uses `u_(s+100)` to emit `y_(s+100)`, then advances. Public
author parity calls use physical inputs `[H,3,B]`, explicit `x0[28,B]` and
`offset=None`, including B=1. Returned author X trajectories are pre-update
states; their last entry is not the final rollout state.

## Conditioning, evidence and full-request cost

Save seed/final context losses, accepted-step and trial counts, termination
reason, scaled gradient, and seed O/final J numerical ranks, singular values
and relative cutoff `1e-12`. Damping makes the step solve identifiable; it does
not make the physical state identifiable. Final J rank is only local and
depends on the active ReLU pattern. Small context residual need not imply small
future error. Save the actual state and check model arrays remain unchanged.

Column scaling reduces simple unit imbalances, but the floors, minimum-norm
seed and damped metric are coordinate dependent. Do not claim invariance under
arbitrary nonorthogonal latent similarity transforms, especially with rank
loss. Keep final learned coordinates and one scale policy for every request;
do not tune damping, iteration count or rank thresholds on DEV scores.

Complete-request timing includes validation, normalization, linear seed solve,
all nonlinear/Jacobian and rejected trial evaluations, diagnostics, final state,
H128 rollout and physical output conversion. Rebuild request workspaces; no
hidden trajectory/state cache across requests. Record actual call counts; a
straight implementation requires at most 17 Jacobian context passes (including
final diagnostics) and 128 trial context passes, plus the linear seed work.
Charge all deployed final matrices/network/statistics, solver constants and
28 retained state scalars. Report transient Jacobian/SVD/workspace separately;
retained `_bla`, factorizations or prepared operators must count if kept by the
deployed implementation. Author archive storage is a separate quantity.

## Fabricated qualification and limits

Before measured use, compare analytic sensitivities against independent central
differences away from ReLU kinks and the pinned author's autodiff, including
nonzero `D_zu,D_yw,B_w`. Test an exact-zero kink separately. Use a known nonlinear
simulator with nonzero initial state to check context residual reduction and
forecast-state indexing; use explicit true state to isolate rollout correctness.
Zero nonlinear feedback must reduce to the qualified linear request. Include
rank-deficient and badly scaled finite systems without claiming their states
are uniquely recovered; test full-rank well-conditioned coordinate transforms
for rollout parity, not finite-budget optimizer invariance. Mutation of dropped
`y_s` or future-input suffixes must not change the estimated context state.
Exercise rejection, stall, cap, failed solve, ownership and B1/B3 public axes.

This makes a bounded causal reference possible; it does not establish that 16
directions find the best nonlinear state. Retain context-fit adequacy diagnostics
and declare optimizer-limited cases. The author's periodic frequency-domain
training objective still differs from this short-context conditional task.
Poor forecast performance alone cannot distinguish weak model learning from
weak state inference, and cannot establish a broad win over the published method.
