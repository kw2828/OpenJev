# Strong-control diagnostic for query-time function reuse

Prospective protocol, 24 September 2026. This is a different task from the
closed observation-reliability studies. No failed criterion or cohort is reused.
The purpose is to qualify a strong conventional control before investing in a
new recurrent architecture. A solved benchmark is not an architectural advance.

## Source and hypothesis

[Dynamic Compression in Recurrent Networks, sections 3-4 and appendix B](https://arxiv.org/html/2608.17896v1)
studies recurrent re-scanning for later function reuse. Its disclosed task uses
eight-dimensional noiseless linear functions, with sixteen labeled examples
and public function IDs in the basis phase. Later four-example queries hide
the function ID. Re-scanning retains the raw prefix as well as the active state.
The main comparisons use selection supervision; a separate codebook method
distills re-scan targets. These are prior methods, not OpenJev inventions.

Our mathematical prediction: Gaussian basis inputs have full column rank almost
surely, so ordinary least squares can recover each function. A query's few-shot
residual identifies the matching function almost surely. This uses the declared
linear/noiseless task structure; it is not learned general-purpose reasoning.
The few-shot-only minimum-norm predictor has expected per-coordinate MSE 0.5
under this distribution (four observed directions out of eight), supplying a
memory sanity control, not a strong competitor with retained history.

## Frozen local task

Independent implementation in NumPy float64, not a reproduction of the paper's
trained models, validation selection, precision, full query schedule or runtime.
Five independent cohorts, each with 64 contexts for each K in {1,3,8,16}.
Every context contains K independent 8x8 standard-Gaussian matrices divided by
sqrt(8), and sixteen standard-Gaussian inputs per matrix. Outputs are x @ A.
For each of eight later requests, sample a function uniformly, supply four
new input/output examples without its ID, and one new unlabeled query input.
No query answer is assimilated. Grouping the basis arrays is equivalent to
using the public basis IDs. Query groups are otherwise independent requests
over the same stored basis, not a trained recurrent sequence.

OpenJev decision extension: each request supplies six independently drawn
Gaussian cost vectors normalized to unit length. The chosen candidate minimizes
the vector's inner product with the predicted output; true regret uses the
actual output. This extension is ours and is not a paper benchmark metric.
Candidate IDs are stable indices 0-5. Ties use the lowest index.

SeedSequence([440260924, cohort, K, case]) spawns six independent streams for
matrices, basis inputs, query identities, few-shot inputs, query inputs and
cost vectors. All 1,280 contexts and 10,240 requests count. No replacements,
checkpoint selection, hyperparameter search, hidden-ID routing or data filtering.

## Methods and accounting

1. Cache per-block ordinary least-squares maps using numpy.linalg.lstsq with
   rcond=None. At each request, select the block with smallest mean squared
   public few-shot residual, then apply its cached map to the query input.
2. Fit minimum-norm least squares to the current four few-shot examples only.

Predictors receive only public basis examples (cached method), current few-shot
examples, query input, and supplied cost vectors for the decision. Hidden
matrices, query identities, targets, seeds and later requests remain outside
the predictor boundary. Rank-deficient and ambiguous fixtures are accepted
with their rank/residuals exposed. Exact ties choose the first public block.
No neural model, optimizer, Astra call or simulator is used.

Report every cohort/K separately: prediction MSE, retrieval accuracy, decision
regret and action agreement, minimum relative singular value, retained ranks,
and complete fitting/query computation. Average all coordinates/queries within
each context then all contexts equally; fixed counts make the flattened mean
identical. No confidence or significance claim is made.

Cached map storage is 64K float64 scalars, versus 256K for raw basis X/Y. Rank
and singular-value diagnostics, temporary fitting arrays, current request
arrays, Python objects and evidence files are additional storage. This is array
accounting, not peak process memory or O(1) total memory. Report diagnostic
bytes separately. Charge all K fits and all K residual evaluations. Fit+query
time includes validation and candidate scoring, excluding generation, file I/O
and reporting. Full process time is reported separately; timings are descriptive
single passes on a shared host, not a production speed or neural-model comparison.

## Fixed interpretation and execution

Classify LINEAR_TASK_SOLVED_BY_CLASSICAL_REFERENCE only if every one of the
twenty cohort/K groups has full basis rank, exact query-ID recovery, prediction
MSE at most 1e-12 and mean decision regret at most 1e-10. Otherwise retain
REFERENCE_NOT_SOLVED. Nonfinite outputs or incomplete work are technical failure,
never an omitted case. This rule assesses task suitability, not model novelty.

Source/runtime/configuration hashes and this protocol are registered before
scientific generation. All three focused test files must pass before launch.
One exclusive run, 120-second worker cap, single-thread numerical libraries.
Independent audit reconstructs maps with QR and few-shot solutions with SVD,
verifies saved-map residual selection exactly, and recomputes every metric from
saved arrays. Numerical comparison tolerance is 1e-10 absolute and relative.
Audit does not replay random generation or reproduce external neural models.
Original process exit and elapsed time are retained. No failed-run retry or
post-outcome threshold change is authorized by this protocol.

If the reference solves the task, do not train a new neural architecture merely
to beat a single-pass GRU here. A future task must specify a meaningful resource
constraint or uncertainty/nonlinearity and retain a strong corresponding
statistical/retrieval control. Recurrent replay already has direct prior art;
neither this reference nor another replay implementation establishes novelty,
connectome benefit, calibrated probabilities, robotics transfer or ICLR readiness.
