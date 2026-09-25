# Causal context initialization for the author linear reference

**Source review and proposed adapter contract only.** No author weights,
measurement arrays, model calls or fits were used in this review. The adapter
qualifies a C100/H128 request interface; it does not reproduce the authors'
periodic warmup or establish author-baseline performance.

## Time convention and independent derivation

Use column-state equations `y_t = C x_t + D u_t` followed by
`x_(t+1) = A x_t + B u_t`. A request beginning at time s provides outputs
`y_s,...,y_(s+C-1)` but only inputs `u_(s+1),...,u_(s+C-1)`. Do not invent
`u_s`, wrap a period, or borrow a future input. Drop `y_s` from this state
solve and estimate the state at time s+1 from the C-1 complete pairs.

Let L=C-1, v_j=u_(s+1+j), z_j=y_(s+1+j), j=0,...,L-1. Construct:

- `f_0=0`; `f_(j+1)=A f_j+B v_j`, the forced state from known inputs.
- `O_j=C A^j`, stacked vertically as a `(L*ny,nx)` matrix.
- `b_j=z_j-C f_j-D v_j`, stacked in time-then-output-channel order.
- `xhat_(s+1)=argmin_x ||O x-b||_2`, with the declared SVD cutoff.

For batched row-state storage, solve all right-hand sides together as
`lstsq(O,b.T)` and transpose the resulting states. The observability recursion
is `O_next=O_current @ A`, not `A @ O_current`.

Advance the estimate through **all L known inputs** to obtain `xhat_(s+C)`.
The first forecast is `C xhat_(s+C)+D u_(s+C)`, then the state advances.
Returning the state before the last context input, or applying the first future
input before computing its output, shifts the request by one sample.

## Rank and coordinate limitations

A fixed `rcond=1e-12` minimum-norm least-squares solve is a defined numerical
policy, not regularization, a stability guarantee or unique physical-state
identification. Preserve rank, singular-value/cutoff diagnostics and residual
information. Zero rank is a legitimate algebraic case: the minimum-norm initial
state is zero, but the known forced trajectory still propagates. Nonfinite
inputs, matrices or results must fail without jitter or clipping.

With full column rank, a nonsingular similarity transform
`A'=T A T^-1, B'=T B, C'=C T^-1, D'=D` preserves the least-squares forecast in
exact arithmetic, and estimated states transform as `x'=T x`.
For rank-deficient systems, the Euclidean minimum-norm representative need not
transform that way under a nonorthogonal T. Predictions remain invariant only
when the discarded state directions cannot affect the forecast outputs.
Finite SVD thresholding can itself change numerical rank under a poorly
conditioned coordinate transform. Do not assert unrestricted state or forecast
invariance for that case.

## Pinned author interface

The pinned [ModelBLA source](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_model_structures.py#L119-L161)
uses the same output-before-state-update convention. Its public
[simulation wrapper](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_model_structures.py#L422-L490)
normalizes physical inputs and denormalizes outputs; explicit x0 is already in
the fitted model's latent coordinates. Normalize context inputs and outputs
with **that model's FIT-only normalizer** before calling the pure initializer.
Do not normalize the latent state or normalize future inputs a second time.

For B requests, call public `simulate` with physical future inputs shaped
`(H,nu,B)`, `x0` shaped `(nx,B)`, and `offset=None`, including B=1. Convert
returned outputs `(H,ny,B)` back to `(B,H,ny)`. The source leaves x0 unreshaped;
its documented 1D-x0/2D-input route therefore needs separate qualification and
is deliberately outside this adapter. Returned X samples are pre-update states;
`X[-1]` is not the state after the final input. Four-dimensional period inputs
and periodic tail warmup are also outside this request contract.

The generic linear initializer must receive no future inputs or targets.
The separate forecast consumes future inputs only after initialization. The
common benchmark scoring normalizer may differ from the author model's own
normalizer; keep physical prediction conversion and scoring normalization
explicit. Charge context normalization, state solve/propagation, forecast,
validation and output conversion in complete-request timing.

## Independent fabricated checks before empirical use

1. Generate observations with a scalar-loop simulator and known nonzero x0,
   nonsymmetric A and nonzero D. Recover the state at s+C and forecast, checking
   explicit first/last input impulses to distinguish pre- and post-update order.
2. Use a well-conditioned nonorthogonal similarity transform on a fully
   observable system. Compare transformed state and physical forecasts, rather
   than comparing implementation intermediates.
3. Use an analytically unobservable coordinate and a zero-observability system.
   Check minimum-norm selection and forced response without requiring recovery
   of unidentifiable state. Include a fixed-cutoff near-rank-deficient witness.
4. Mutate only the dropped first output and separately future-input suffixes.
   State estimation must ignore both future inputs and the dropped output;
   forecast prefix causality must hold. Requests and chunked states must not alias.
5. Compare the pinned author's public three-dimensional-input simulation, with
   explicit estimated x0, against the independent simulator. Match both outputs
   and pre-update state trajectories; keep this distinct from a numerical fit.

These checks validate a causal linear interface. A nonlinear-LFR context state
solve remains a separate design and qualification task.
