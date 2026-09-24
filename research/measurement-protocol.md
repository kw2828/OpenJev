# Consolidating observations in bounded recurrent memory

Prospective protocol, 24 September 2026, `measurement-v1`.
The [learned deletion study](retention-results.md) remains failed. This is a
separate experiment with fresh fields, no inherited weights and no training.

## Question and limits

Can a fixed linear sketch of all observations beat retaining selected raw
observations at the same 1,024-byte logical state cap? The primary candidate
is a spectral sketch. DCT and spatial-bin sketches are alternative analytic
controls, not selectable replacements for the primary candidate.

This uses established Gaussian conditioning on projected observations, closely
related to [computation-aware GPs, Sections 3-4](https://papers.nips.cc/paper/2024/file/379ea6eb0faad176b570c2e26d58ff2b-Paper-Conference.pdf).
It is not a new GP, trained architecture, connectome model or learned world
transition. [WISKI](https://proceedings.mlr.press/v130/stanton21a.html) is related
streaming sufficient-statistic work, but changes the kernel approximation.
Here the original kernel is retained and information is lost through projection.

## Unchanged world, fresh data

Use the original known-law 17 by 17 grid in [-2,2]^2 at spacing 0.25, with
zero-mean RBF field, length/amplitude 1, spatial white variance 1e-5 and
independent observation-noise variance 0.09. Each stream has unique locations.
The grid restriction is public information shared by every method.

After the stream, four requests each offer four paths of four latent points.
Exposure is the path's mean, with no extra observation noise. Path cost is
0.02 plus the probability that exposure exceeds 0.5; defer costs 0.20.
Choose the first minimum in path order then defer. Axial and diagonal paths
have the same point count and index stride; diagonal paths are longer.

Use the unchanged `retention_data.generate` algorithm with namespace 553260924.
Three cohorts of 128 fields per population, four requests per field:

| Population | Seed offset plus cohort 0,1,2 | Observations | Geometry |
| --- | ---: | ---: | --- |
| BASE | 100 | 64 | axial |
| SHIFT | 200 | 64 | diagonal |
| LONG | 300 | 192 | axial |
| LONG_SHIFT | 400 | 192 | diagonal |

Total 1,536 independent fields, 6,144 requests, 24,576 latent path outcomes.
Within-field requests/outcomes are correlated. No difficulty/outcome filtering.

## Causal memory and prediction

All hybrid sketches retain a 37-byte little-bit-order occupancy mask and 118
float64 values. Through observation 118, values are raw labels sorted by grid
ID. At observation 119, transform all 119 labels into 118 linear sums exactly
once. Later writes add `a(x) * y`. The step counter determines representation.
No discarded label, basis table, factor, prediction or RNG state is retained.
Only new coordinates/labels enter writes, never future paths or private outcomes.

The spectral basis consists of the 118 leading separable eigenvectors of the
17-point one-dimensional RBF matrix. Canonicalize each vector's sign by making
its largest absolute entry positive. Sort product modes by descending eigenvalue
product, breaking ties lexicographically by mode indices. Uniform spatial white
variance leaves those eigenvectors unchanged. DCT uses orthonormal DCT-II product
modes ordered by i^2+j^2 then i,j. Bins assign grid ID to floor(ID*118/289), with
unit coefficients. Regenerate every basis as scratch, including in timings.

After compression, reconstruct observed coordinates from the mask. For the
restricted basis A, compute thin SVD A=U diag(s) Vt. Retain singular values
strictly greater than float64 epsilon times max(A.shape) times max(s). Use
t=U_retained^T z/s and H=Vt_retained. Condition the original GP on t=H y:
`C=H (K_XX + 0.09 I) H^T`, with query cross-covariance `K_*X H^T`.
This is exact conditioning on the retained numerical row space, not a claim
to preserve every nominal sum when numerical rank is reduced. Record rank and
cutoff. Preserve shared-coordinate covariance and independent event noise.
No covariance clipping, adaptive jitter, or replacement contexts are permitted.

## Fair controls and resource accounting

The known finite grid also helps raw storage, so the old 41-row float-coordinate
buffer is insufficient as the principal control. PackedRaw118 uses the same
mask and 118 raw labels in sorted grid order. On overflow, remove the point
with smallest nearest-neighbor squared distance; exact ties remove lowest grid
ID. Recent98 stores ordered uint16 grid IDs and float64 labels. Coverage98 uses the same packed row encoding as Recent98, with nearest-neighbor
deletion and the oldest-retained tie convention. This controls for the possible
spatial bias of canonical grid-ID ties.
All raw controls use exact GP conditioning on retained observations.

| Method | Persistent numeric state | Logical bytes/context |
| --- | --- | ---: |
| spectral118, dct118, bins118 | mask, 118 values, four floats, step, one-byte kind | 1,022 |
| coverage118 | mask, 118 values, four floats, step | 1,021 |
| recent98 | 98 uint16 IDs, 98 values, four floats, step | 1,020 |
| coverage98 | 98 ordered uint16 IDs, 98 values, metadata | 1,020 |
| full GP | all original coordinate/label triples, metadata | 1,576 / 4,648 |

BASE/SHIFT are exact-prefix sanity checks for the hybrids and packed controls,
not compression wins. LONG/LONG_SHIFT test actual compression. The full GP
exceeds the cap but provides the conditional decision reference. Logical state
excludes shared program code, Python object overhead and temporary workspace.
Report traced Python/NumPy peak allocation; native allocations and RSS remain
unmeasured. Do not claim equal peak memory or an asymptotic storage result for
unbounded domains: the finite-grid mask is essential here.

## Frozen assessment

Report per-cohort and population decision regret against full-public-history
GP risk, latent-exposure Gaussian NLL, MSE, Brier score, 90% coverage, defer rate
and risk error. Private realized outcomes only score predictions, never choose
actions. Full-GP zero regret is definitional. Preserve all seven methods.

The spectral primary must pass all 26 conditions:

- BASE/SHIFT regret at most 1e-8 in every cohort: six conditions.
- LONG/LONG_SHIFT mean regret at least 30% below each of coverage118, recent98
  and coverage98, with only 1e-6 absolute tolerance: six conditions.
- Each population mean NLL no more than 0.02 nat above the best raw control:
  four conditions.
- Every LONG/LONG_SHIFT cohort regret at least 10% below its best raw control,
  with 1e-6 absolute tolerance: six conditions.
- Both long populations beat always defer: two conditions.
- On each long population, the upper 95% paired-bootstrap confidence limit
  of spectral-minus-coverage98 regret is strictly below zero: two conditions.

Bootstrap uses 1,000 resamples of the 384 whole-field mean regrets, preserving
within-field queries; NumPy default_rng seeds namespace+900 and namespace+901.
Use 2.5/97.5 percentiles with linear interpolation. This comparison is prespecified;
the interval does not support claims outside this task or a tuned model search.

Pass yields `ADVANCE_CONSOLIDATION_BASELINE`, otherwise
`DO_NOT_ADVANCE_CONSOLIDATION`. A pass qualifies a memory baseline for subsequent
learning research. It establishes neither superiority over the DCT control nor
architectural novelty. Never promote a different sketch after primary failure.

## Execution and evidence

One CPU thread, float64, 1,200-second run cap. No training or empirical retries.
Freeze protocol, implementation, tests and independent auditor before generation.
Use fabricated deterministic cases for dense conditioning, transition correctness,
rank deficiency, state ownership, bytes, causal interfaces, metrics and gates.
Save all 12 datasets, 84 prediction groups, 72 states and 28 resource records.
Independent audit reconstructs states and projected conditioning from saved
public inputs using separate formulas, at rtol=atol=1e-8. It never regenerates
private fields. Timings and historical execution remain original receipt evidence.

Timing includes every stream write and four queries, basis reconstruction and
factorization: first field of each population, three warmups, ten repetitions.
This is a descriptive timing probe, not a population speed estimate. Preserve
partial outputs and the original failed or timed-out process; do not retry.
