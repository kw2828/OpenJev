# Pooled FIT-only author BLA28 runtime contract

**SOURCE-VERIFIED CONTRACT; PROPOSED QUALIFICATION, NOT EXECUTED HERE.** This note reads code and notebook source only. It does not load measurements, pretrained models or notebook outputs, install dependencies, or report a fitted author baseline. The purpose is a fair author-method reference, not another model architecture.

## Immutable sources

- Author implementation: [`freq-statespace` a79e8c567b018a6c9462528fc1e10b77fd19b3e2](https://github.com/merijnfloren/freq-statespace/tree/a79e8c567b018a6c9462528fc1e10b77fd19b3e2), package version 0.1.2.
- FSM recipe: [`fit_BLA.ipynb` at 539a12fef384b086a8562b500498b2fa3899ef70](https://github.com/merijnfloren/fsm-benchmark-data/blob/539a12fef384b086a8562b500498b2fa3899ef70/baseline_results/fit_BLA.ipynb).
- Licensing and scope distinctions remain in [author-reference qualification](fsm-author-reference-qualification.md): GPLv3-or-later implementation, CC BY 4.0 dataset. The two repository pins do not reconstruct the historical author environment automatically.

## Verified API and array contract

The top-level import is `import freq_statespace as fss`. `fss.lin` exposes the BLA functions; `fss.ModelBLA`, `fss.Normalizer`, `fss.save_model` and `fss.load_model` are public exports. [`__init__.py`](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/__init__.py).

| Public entry point | Exact arguments/defaults relevant to this run |
|---|---|
| `fss.create_data_object(u, y, f_idx, fs)` | Four positional arguments; returns `InputOutputData`. |
| `fss.lin.subspace_id(data, nx, nq=None, freq_weighting=True, input_output_mode=False, logging_enabled=True)` | `nq=None` resolves to `nx+1`; explicit `nq` must exceed `nx`. Returns `ModelBLA`. |
| `fss.lin.optimize(model, data, *, solver, freq_weighting=True, input_output_mode=False, max_iter=1000, print_every=1, return_solve_details=False, device=None)` | Actual default `solver` is `optimistix.LevenbergMarquardt(rtol=1e-3, atol=1e-6)`. With `return_solve_details=True`, returns `(model, SolveResult)`. |
| `model.simulate(u, *, x0=None, offset=None)` | Returns `(y, t, x)` in physical output units and internal state coordinates. |
| `fss.save_model(model, path)` / `fss.load_model(path)` | ZIP checkpoint; loading returns the model, not optimizer state. |

Signatures and defaults: [`_data_manager.py`](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_data_manager.py#L128-L228), [`_best_linear_approximation.py`](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_best_linear_approximation.py#L78-L231), [`_config.py`](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_config.py).

Input axes are **sample, channel, realization, period**: `u[N,nu,R,P]`, `y[N,ny,R,P]`. The proposed real FIT tensor has `N=8192`, `nu=ny=3`, `R=6`, `P=2`: take realizations 0..2 at 100 mV, then 0..2 at 200 mV, concatenating on axis 2. Do not flatten periods into time or split an orthogonal triplet. Set `fs=6400.0` and `f_idx=np.arange(1,3840)`, matching the notebook's frequencies below 3,000 Hz with DC excluded.

`data.norm` contains owned channel statistics named `u_mean`, `u_std`, `y_mean`, `y_std`, each length three. They are computed over axes `(0,2,3)` with population standard deviation before period averaging. `data.time.u/y` then have shape `[N,3,R]`; `data.freq.U/Y` have shape `[N//2+1,3,R]`. `data.freq.G_bla.G` is `[len(f_idx),3,3]`. Period noise estimates and between-triplet variance are diagnostics, not a reason to enable weighting across different amplitudes. Use **explicit `freq_weighting=False`** for initialization and refinement, as in the author's pooled-amplitude recipe. Wrapper guards must add finite values, positive standard deviations, valid frequencies and exact roster checks; the upstream constructor alone does not enforce all of them.

The returned `ModelBLA` has `A[nx,nx]`, `B_u[nx,3]`, `C_y[3,nx]`, `D_yu[3,3]`, scalar `ts=1/fs`, and `norm`. For order 28, `num_parameters()` returns 961 matrix entries; it excludes normalization and request state. The update is **output first**: `y_k=C_y*x_k+D_yu*u_k`, then `x_(k+1)=A*x_k+B_u*u_k`. A returned state trajectory contains pre-update states, so its last row is not the final post-transition state. [`ModelBLA`](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_model_structures.py#L17-L213).

For public simulation, preserve an explicit realization axis even for one request: `u[H,3,B]`, `x0[nx,B]`. The wrapper normalizes raw inputs and denormalizes outputs. Do not rely on the documented two-dimensional-input/one-dimensional-state shortcut before qualifying it: source passes a supplied `x0` through unchanged, while the internal loop assumes a matrix. Set `offset=None`; a positive offset prepends tail inputs and is only appropriate for periodic simulation. [`simulation wrapper`](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_model_structures.py#L460-L582), [`tail extension`](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_misc.py#L62-L86).

## CPU float64 and the eventual FIT call

Start a fresh isolated process with `JAX_PLATFORMS=cpu` and `JAX_ENABLE_X64=True` before importing JAX or the author package. Set `jax.config.update("jax_enable_x64", True)` before array creation as an explicit assertion of intent; verify the backend is CPU and real/complex arrays are float64/complex128. Use `device="cpu"` in `optimize` and a CPU default-device context around creation/subspace calls, which have no device argument. These configuration controls are documented by [JAX](https://docs.jax.dev/en/latest/config_options.html#platforms). Pin the resolved JAX/JAXlib, NumPy, SciPy, Equinox, Optimistix, Optax, Jaxtyping and loader versions; record BLAS/thread settings separately. Do not infer single-thread execution from the CPU setting.

The following is the proposed real-data call sequence after independent data admission and qualification, not code executed by this note:

```python
data = fss.create_data_object(u_fit, y_fit, np.arange(1, 3840), 6400.0)
initial = fss.lin.subspace_id(
    data, nx=28, nq=29, freq_weighting=False,
    input_output_mode=False, logging_enabled=False,
)
model, solve = fss.lin.optimize(
    initial, data, solver=optx.BFGS(rtol=1e-3, atol=1e-5),
    freq_weighting=False, input_output_mode=False,
    max_iter=5000, print_every=-1, return_solve_details=True, device="cpu",
)
```

This preserves the notebook's order, subspace initialization and BFGS settings while restricting its data scope. `input_output_mode=False` fits the nonparametric frequency response rather than individual output spectra. State-coordinate normalization occurs before and after refinement; compare transfer functions/forecasts rather than requiring matrix entries to match a generative realization. The library may stop before the iteration cap; record its raw stop flag and the iteration limit separately, not just process exit.

`SolveResult` fields are `theta`, `aux`, `loss_history`, `iter_count`, `iter_times`, `converged`, `wall_time`. It is not a restartable BFGS checkpoint. The solver uses private Optimistix helpers and performs a warm-up step before its reported wall-clock interval; iteration timing does not by itself establish end-to-end CPU cost. Preserve an outer process clock and synchronize final arrays before closing it. A source-level edge case also needs a retained fixture: `aux` is assigned inside the update loop, so a zero-update route may fail at return. BFGS initializes its stop flag to false, so this is not an ordinary initially converged BFGS route; `max_iter=0` or another solver can expose it. This is an unexecuted source finding. [`_solve.py`](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_solve.py#L21-L169).

**Stop status is not a convergence certificate.** The author loop stores only `solver.terminate(...)[0]` as `converged`, discarding the result code. Optimistix 0.1.0's generic driver stops on either its boolean or an unsuccessful result and checks parameter finiteness; Newton/Chord can explicitly signal divergence through a true boolean. For the exact default inverse-Hessian BFGS used here, the boolean is an accepted-step Cauchy small-change test, while its Armijo search and inverse-Hessian descent return successful status without numerical-failure detection. Report `author_stop_flag` or “BFGS Cauchy stopping condition met,” and independently check final finite parameters and loss. This does not establish a small gradient, an optimum, stable dynamics, or successful generic-solver status. The unmodified return object cannot recover the discarded status. Sources: [generic iteration](https://github.com/patrick-kidger/optimistix/blob/v0.1.0/optimistix/_iterate.py#L214-L258), [BFGS](https://github.com/patrick-kidger/optimistix/blob/v0.1.0/optimistix/_solver/quasi_newton.py#L210-L311), [Armijo](https://github.com/patrick-kidger/optimistix/blob/v0.1.0/optimistix/_solver/backtracking.py#L55-L106), [inverse-Hessian descent](https://github.com/patrick-kidger/optimistix/blob/v0.1.0/optimistix/_solver/gauss_newton.py#L35-L78), [Newton/Chord](https://github.com/patrick-kidger/optimistix/blob/v0.1.0/optimistix/_solver/newton_chord.py#L154-L196).

## Proposed fabricated qualification system

Use one deterministic, observable/controllable order-4, three-input/three-output stable system. It qualifies numerical plumbing only; the empirical reference remains order 28. Define, in float64:

```text
A = diag(1/5, 2/5, 3/5, 4/5)
B = [[1, 0, 1/4], [0, 1, -1/4], [1/2, 1/4, 1], [1/4, -1/2, 1/2]]
C = [[1, 1/5, 0, 1/4], [0, 1, 3/10, -1/4], [1/4, 0, 1, 1/2]]
D = diag(1/10, 1/5, 3/10)
N = 256; fs = 256.0; excited k = 1..63; R = 6; P = 2
```

Construct two orthogonal triplets directly in the RFFT domain, with zero DC/Nyquist and zeros outside the excited frequencies. For block `m=0,1`, input channel `c=0,1,2`, and triplet column `r=0,1,2`, set

`U[k,c,3*m+r] = (m+1) * exp(2*pi*i*k*(c+1)*(m+1)/101) * exp(-2*pi*i*c*r/3)`.

Each excited input block is a scaled unitary DFT matrix. Independently form `G_k=C*(exp(2*pi*i*k/N)*I-A)^(-1)*B+D`, set `Y_k=G_k*U_k`, then real inverse FFT along the sample axis. Duplicate the exact period into the last axis. This creates steady-state data without transient removal or random sampling. Use no measurement noise and unweighted fitting, so zero variance estimates are never inverted.

Freeze this fixture before its first execution. Proposed checks:

1. `create_data_object` yields the exact axes/statistic fields above; independently verify its normalized frequency response against `diag(1/y_std)*G_k*diag(u_std)`. Subspace identification uses `nx=4,nq=5`, explicitly unweighted, and should recover the transfer function at fixed `rtol=atol=1e-6` without comparing coordinate-dependent matrices.
2. To exercise a genuine nonzero refinement loss, make an isolated copy of the initialized model with `D_yu` increased by `0.02*I3`; preserve the unperturbed model separately. Run the same BFGS tolerances with a fixed **25-iteration qualification cap**. Require finite model/history, at least one recorded iteration, and independently recomputed unweighted frequency-response error no worse than the perturbed start. Record the raw stop flag separately; a 25-step smoke is not convergence qualification for order 28. Preserve failure instead of changing fixture, tolerances or solver until it passes.
3. Compare explicit-axis public simulation against a standalone NumPy output-before-update loop at `rtol=atol=1e-10`, including one and multiple requests, nonzero initial state and direct feedthrough. Verify returned pre-update states and calculate the final state explicitly. Use this to qualify the causal adapter, not periodic tail warmup.
4. Round-trip the locally generated checkpoint, checking all matrix/statistic/scalar fields and forecasts. Include shape/finite guards, read-only input ownership and a no-call spy for the eager benchmark loader. None of these tests may open the FSM archive or an author-supplied model.

## Serialization and causal-request join

`save_model` writes `config.json` plus `weights.eqx`, adds `.zip` only when no suffix was provided, creates parent directories and **overwrites an existing file**. Our wrapper should allocate an exclusive output directory and reject an existing checkpoint before calling it. Save initial and final ZIPs, hashes and solver metadata separately. For independent deployment/audit, also export explicit `A/B_u/C_y/D_yu/ts/u_mean/u_std/y_mean/y_std` arrays in a documented numeric-only format, with raw-byte provenance. Do not expect `load_model` to recreate optimizer state. [`_serialize.py`](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_serialize.py#L95-L153).

The separate NumPy causal adapter accepts the exported matrices. Its C100 initializer estimates the state at the second observed sample from 99 paired observations/inputs, using `O_j=C*A^j`, subtracting forced response and direct feedthrough, with fixed `rcond=1e-12` and rank diagnostics. Propagate those 99 known inputs to the first forecast state, then emit `C*x+D*u` before each update. Normalize with the author FIT statistics, and use the existing study's fixed FIT statistics only for final error scoring if the two normalizers are stored separately. No unavailable first input, future output, periodic tail or extra period is admitted during conditioning.

The smoke establishes environment and interface behavior, not performance, stability certification or author-score reproduction. BLA28 remains a FIT-only refit, and the fixed C100/H128 evaluator remains our conditional forecasting task. Author NL-LFR qualification can follow using the same admitted data and causal interface; it is not silently replaced by this smaller fabricated system.
