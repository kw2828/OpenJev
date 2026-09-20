# Typed decisions and rare-category support

This new development experiment follows the [closed conditional-observation
study](dialogue-conditional-results.md) and its [saved-output error diagnosis](dialogue-conditional-error-results.md).
It asks whether separating answer-type mass from candidate competition helps
beyond simply increasing the training weight of rare answer types. It does not
resume a closed fit or test a recurrent model, proprietary RLCD, biological
wiring, uncertainty calibration or an architecture-novelty claim.

## Fixed comparison

| Output normalization | Existing stratum weights | Rare-category weights |
|---|---|---|
| Flat | `flat_stratum` | `flat_balanced` |
| Typed | `typed_stratum` | `typed_balanced` |

All four arms use the existing frozen MiniLM candidate-conditioned token
attention, query/candidate embeddings, ten public lexical features and the
correct previous categorical value. All also receive the same five public
candidate-type flags: NOT_MENTIONED, DONTCARE, TRUE, FALSE and OTHER. Flags
describe every schema candidate and never encode the current gold target's
type. Actor construction ignores current labels, transition classes and
evaluation panels. Correct previous values remain a privileged input, not an
autonomous memory claim. The inherited lexical features can contain causal
literal-register history.

Both heads construct the same candidate hidden representations `h_c`, scalar
candidate scores `z_c`, and three branch gates `g_b`. Branches are
NOT_MENTIONED, DONTCARE and concrete values. Each gate is a shared linear map
of the mean candidate hidden representation in its branch. Padding is excluded.
Every question has exactly one NOT_MENTIONED and one DONTCARE candidate plus
at least one concrete candidate.

- Flat: `log p(c) = log_softmax(z_c + g_branch(c))`.
- Typed: `log p(c) = log_softmax(g)_branch(c) + log_softmax(z within branch)_c`.

Flat branch mass therefore depends on `g_b + logsumexp(z within b)`; typed
branch mass depends only on `g_b`. This intervention removes within-branch
score scale and cardinality from branch competition. It supplies no additional
semantic evidence. TRUE/FALSE remain separate candidate types within the
concrete branch; NOT_MENTIONED and DONTCARE are resulting values, not speech
acts or privileged operation labels.

Default width is 384 input, 64 projections/hidden units and 64 attention units.
Both arms have **173,506 registered parameters**. This is not equal effective
capacity: singleton within-branch scores cancel in typed normalization, scalar
head/gate biases are common shifts, and the entropy-input column is zero.
Both compute all shared scores, but normalization work differs and its cost
is included. This is a conventional factorization diagnostic, not a new
language representation.

## Prospective split inside official training

The rule was selected before examining membership: take the first `ceil(20%)`
of all original TRAIN services sorted by
`SHA256("openjev-typed-v1:" + exact_service_name)`, breaking ties by name.
Use original public service lists, including services without categorical
admitted rows. Exclude **every dialogue containing any selected service** from
fitting. Evaluate all admitted rows in those dialogues. The primary panel
contains only rows whose supplied query service is selected; other services
in these dialogues are a secondary panel.

The [metadata-only split](../output/dialogue-typed-v1/split-design-01/summary.json)
selects Events_1, Homes_1, Hotels_1, Music_2, Services_1 and Services_3 from 26
services across 16,142 original dialogues. It was checked against all 127
original TRAIN shards. No alternative split was searched. The model runner
authenticates the saved original-service projection and recomputes membership.

| Partition | Rows | Dialogues | Schema queries | Changed / retained |
|---|---:|---:|---:|---:|
| Fit | 29,211 | 1,343 | 41 | 2,299 / 26,912 |
| Evaluation | 13,599 | 674 | 51 | 1,048 / 12,551 |
| Primary selected services | 7,819 | 674 | 12 | 578 / 7,241 |
| Secondary other services | 5,780 | 400 | 39 | 470 / 5,310 |

The primary changes contain TRUE 29, DONTCARE 5, OTHER 544, FALSE 0 and clears
0. DONTCARE covers only one schema query and remains weak descriptive evidence.
This cannot evaluate unseen FALSE/clearing transfer or general Boolean polarity.
Services can share domains with fitted services; this is not domain-held-out
evaluation. All rows appeared in earlier experiments. This is an internal
development split, **not historically untouched confirmation**.

Reuse the authenticated metadata/cache closure from the conditional study,
preparation completion `960afa60172056134bc4d3cc523338b5fdb8886b55e2ef41e8ce125c72dffbb3`.
The split receipt is
`fca0db6a135eec3c845fe2b8bc5db151c4acdf0bb4b669ed1ff9ca42b7287118`.
Mixed preparation metadata contains official DEV rows and is authenticated as
a whole; those rows do not enter this experiment's fitting or evaluation.
Official test remains untouched. No new encoder or teacher calls.

## Training and pairing

Train all four arms at seeds **6101, 6102 and 6103**, exactly twelve fresh final
fits. Copy the full initial state across all four arms within each seed and
record its hash. Never load earlier fitted weights. Use 20 complete, shared
NumPy permutations of fit rows per seed, batch size 256 including the final
short batch, AdamW learning rate 0.001, weight decay 0.0001 and gradient-norm
clip 1.0. No checkpoint selection, schedule or early stopping. CPU float32,
four intra-op threads, one inter-op thread and deterministic Torch algorithms.

Visit methods in table order at 6101, rotated one position at 6102 and two at
6103. Each fit performs **2,300 updates**, or **27,600 total**. Each evaluates
13,599 rows in 54 batches in canonical order. Timing includes any ordering
effects; this is not a balanced four-period inference-latency experiment.

Keep the same row sampling for both objectives. Training strata are
unmentioned retention, assigned retention and changed state. For `N` fit rows,
stratum count `n_s`, and nonempty target-type count `k_s`:

- Stratum weights: `N / (3 * n_s)`.
- Balanced weights: `N / (3 * k_s * n_(s,type))`.

Compute weights from fit labels only. Each recipe has mean weight one and
gives each stratum total weight `N/3`; balancing additionally equalizes its
nonempty target categories. Empty cells remain unsupported, not fabricated.
Use the ordinary batch mean of weighted target NLL, without batchwise weight
renormalization. Evaluation is unweighted.

## Fixed analysis and decision

Save all final distributions and weights before quality analysis. Report all
twelve fits, every seed and per-service effects. For selected-service,
other-service and combined panels, report accuracy, direct-log NLL and
multiclass Brier, including changed/retained, five transition groups and target
value groups. Preserve row, dialogue, service and schema-query support; show
row-weighted, equal-dialogue and equal-service means. Empty means are undefined.
Three seeds are repeat optimizations, not three independent service samples.

The single mechanism contrast is **typed_balanced versus flat_balanced** on
the selected-service panel. Continue this mechanism only when:

1. Three-seed mean equal-service changed-state NLL falls by at least 5%, with
   no paired seed worse.
2. TRUE-change and DONTCARE-change top-1 recall each improve by at least five
   percentage points, averaged across the three fits on their common rows.
3. Each type's false-positive rate and overall retained-state error rise by
   at most 0.5 percentage points. TRUE false positives use non-TRUE targets
   only where TRUE is a valid candidate; DONTCARE uses its supported non-target
   rows. These are row rates averaged across seeds on the selected services.

Require nonempty denominators and all twelve complete, numerically valid fits.
These are prospective engineering relevance thresholds, not statistical
significance or population guarantees. The five-case DONTCARE panel cannot
support a broad generalization claim even if the screen passes.

Report flat_balanced versus flat_stratum, typed_stratum versus flat_stratum,
and their factorial interaction as explanatory contrasts. They cannot replace
the mechanism contrast. A weighting-only gain supports an optimization/support
explanation. A loss reduction without repaired rare decisions, or more rare
predictions accompanied by excess false writes, does not earn continuation.
Failure closes this normalization recipe; a pass only motivates an independently
justified broader experiment. Neither outcome changes the earlier failed study.

Include previous-gold carry and the unchanged literal-register carry as
accuracy-only references. A third reference scores each public candidate type
by its fit target count plus one, distributing that type's score uniformly
over candidates of that type, then chooses the first maximum. It uses no
evaluation labels. These deterministic choices receive no finite loss score.

## Execution and evidence

Run synthetic model, actor/split/objective, complete small training-loop and
reporter checks, then publish code, protocol, exact split/weights/row orders and
source snapshots before one scientific invocation. Use an exclusive output
directory. Preserve every update's rows, objective, probability checks and work
counts, all twelve final checkpoints/distributions, exact initialization hashes,
runtime, whole-run and per-fit elapsed times and sampled process-lifetime RSS.
Reject nonfinite losses/gradients, invalid masks, missing rows and unnormalized
supported distributions. No probability floors or repairs.

The whole run has ceilings of **3,600 seconds, 6 GiB RSS and 512 MiB output**.
Charge authentication, gathering, all fitting/evaluation, bookkeeping, I/O and
payload hashes. Completion writing and return remain cap-checked. Per-fit
timing includes its optimizer setup, training, evaluation and saved payloads;
shared cache loading and seed initialization remain whole-run costs. Inherited
feature preparation and separate preflight/reporting costs are not free and
must be attributed separately. No inference speedup claim follows from fit time.

Coordinate a quiet local CPU window before starting. On failure preserve final
and partial artifacts, returned prediction prefixes and the original error.
No retry, resumption, replacement seed or cap extension under this protocol;
do not score partial fits as a scientific comparison. Report from saved outputs
only after complete execution, with an independent numerical check. Publish
negative results and every relevant regression.
