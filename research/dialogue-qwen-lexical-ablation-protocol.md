# Qwen lexical-input ablation

Prospective specification, 20 September 2026. This follows the
[completed retention diagnosis](dialogue-qwen-retention-results.md), before
preparation, pilot or inference for this experiment. Its purpose is to test a
cleaner observation baseline before adding recurrent memory. It is not a new
architecture experiment or a fresh confirmatory evaluation.

## One intervention, complete coverage

Compare the original Qwen scorer with the same scorer after removing the ten
lexical flags and their explanatory text. Apply this intervention to both
`current` and `history4`, preserving all 7,819 rows per arm, 578 changed and
7,241 retained. Every original model prediction remains unchanged and is reused
as the paired reference. Do not select rows from the twelve reviewed cases,
correctness, probability, type, service or any new output.

Use only the authenticated original actor requests, without reading baseline
scores during preparation. The original preparation plan is
`2d5f7e6b512ae7260cc01685ae03220891ad4ca5236092645197d4074be90111`,
and its completion receipt is
`6a7a8283efa612866a0a9f0c2bcce92bb54e9ce8982bde26baaa8d8aeb24eb9f`.
Verify the exact preparation directory membership and every payload descriptor
before decoding requests. Copy the evaluator-label bytes without decoding them.
Do not read official DEV or TEST or add examples.

The transformation has exactly three anchored removals:

1. Remove each candidate description's trailing `Public lexical flags` line,
   including its preceding newline and ten binary comma-separated values.
2. Remove the two exact TASK sentences beginning `Lexical flags are noisy` and
   `The literal register carries`, including their original trailing spaces.
3. Remove the question's trailing `Lexical flag order` line and preceding newline.

Reject malformed or nonmatching source packets. Preserve every other character
of question and candidate text, candidate types and IDs, raw dialogue context,
role boundaries, supplied previous value, canonical mapping, row and request
identities, grouping and processing order. Verify that the deterministic
description sort maps the same canonical IDs to the same A-L token labels.
Do not introduce KEEP/SET wording, examples, schema-specific rules, confidence
thresholds, new candidate choices or additional inference steps.

Re-tokenize the changed prompts using the same local pinned tokenizer and
chat template. Preserve the original four-question maximum batch and full-prompt
`batch` scoring method. Enforce every existing character and token limit without
truncation or row removal. Record exact tokens and padded workload counts.

## Identity and execution

Use the unchanged model `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, precision, runtime, candidate
readout, temperature one and frozen prompt-order tie rule. Authenticate all
eleven cached model files. No model substitution, download, fitting, generation,
calibration, prompt search or checkpoint selection is allowed.

The original [runner](../scripts/run_dialogue_qwen_observation.py) remains
unmodified. The plan's `version: dialogue-qwen-observation-v1` denotes its
compatible execution schema, not the scientific study identity. The new plan
must additionally contain `experiment_id: dialogue-qwen-lexical-ablation-v1`,
the original preparation pins, the new protocol hash and the complete source
closure. Preserve the original nine source bindings and append the transformation
module, preparation script, their two tests and this protocol. Preserve the
original model/runtime/input descriptors and label bytes. Recompute only the
changed request payload, token counts and prescribed pilot selection.

All phases use new exclusive output directories, one CPU thread, start and
completion/failure receipts, and whole-phase caps:

| Phase | Time | Peak process RSS | Output |
| --- | ---: | ---: | ---: |
| Preparation, local tokenizer only | 180 seconds | 2 GiB | 512 MiB |
| Fresh cost pilot | 300 seconds | 12 GiB | 512 MiB |
| Complete inference | 7,200 seconds | 12 GiB | 512 MiB |
| Saved-output report | 60 seconds | 2 GiB | 64 MiB |

Preparation initializes no model. Synthetic transformation and authentication
checks precede it. Source review and exact paired-input verification precede
the new pilot. Acquire the shared numerical allocation before model loading.

Use a **fresh pilot bound to the new plan**. Select the union of the first twelve
canonical dialogue/time groups and each arm's longest-token group, including
all corresponding question chunks, exactly as the unchanged runner specifies.
The pilot discards every quality-bearing output and saves only numerical checks,
identities, work and timing. It never accesses evaluator labels.

Reuse the original admission formula, not its pilot measurements. For each arm,
take the maximum observed pilot seconds per padded token slot, multiply by that
arm's complete padded workload, sum, multiply by two, then add whole-pilot wall
time and 60 seconds and round upward. Admission requires at most 7,200 seconds
and every numerical, identity, coverage and resource check. This is a heuristic
cost projection, not a timing guarantee.

If admitted, execute all 15,638 decisions once. Charge preparation, pilot and
complete inference separately and together. The repeated pilot rows are paid
cost, not a quality replicate. Preserve attempted/returned call counts, exact
request order, raw candidate logits, log probabilities, vocabulary-mass witnesses,
timings and memory. Do not inspect partial quality. A failed preparation, denied
pilot or failed full run ends this version: preserve its files, with no retry,
resume, cohort shrinkage, alternative prompt or budget extension.

## Paired saved-output comparison

The original completed run pin is
`872ee6819af4cbc4f7e0bd6397d6c90907dba9aaecc995abb6045fd20520f70c`;
its summary pin is
`931f60349dc7dace7508d0f3307a6c480c4e14022de7e2c047b6f024ec349c6c`.
Authenticate both complete chains before labels or predictions are decoded.
Check byte-identical evaluator labels, complete row/candidate coverage and the
exact permitted public-input transformation. The existing reporter's exact
source closure and scientific rules cannot serve as this experiment's report
unchanged. A separate reporter may reuse its pinned reconstruction and metric
functions, explicitly identifying that inherited verification.

Reconstruct float64 supported-label softmax and NLL without floors. Preserve
every exact tie and the original first-label selection. Report paired repairs
and harms for all, changed, retained, unmentioned-retention and assigned-retention
rows, by service and candidate type. Report accuracy/error, NLL and multiclass
Brier under row and equal-service weighting, with undefined rates preserved.
Show every service and rare-type limit, including no FALSE targets. No threshold,
calibrator, model combination or label correction is fitted.

The development continuation rule requires **all sixteen components** below:
eight per context arm. Let each delta be `no-flags minus original`, calculated
from full-precision values with inclusive boundaries.

| Metric | Row-weighted requirement | Equal-service requirement |
| --- | ---: | ---: |
| Retained error | Delta ≤ -0.02 | Delta ≤ -0.02 |
| Changed accuracy | Delta ≥ -0.01 | Delta ≥ -0.01 |
| Overall NLL | Delta ≤ 0 | Delta ≤ 0 |
| Overall Brier | Delta ≤ 0 | Delta ≤ 0 |

Equal-service means average the same supported services within each stratum.
Display all sixteen decisions individually, both per-arm conjunctions and their
joint result. A pass promotes the cleaner input representation for subsequent
development only. It neither reverses an earlier failed gate nor proves an
architecture, calibration, benchmark or statistical-significance claim.

Record all costs and token reductions. Because the baseline was executed earlier,
its timing comparison is descriptive, not an interleaved speed benchmark. Do not
credit removal of text as a novel inference algorithm. The historical small-model
results may remain contextual references, never a replacement paired comparator.

## Interpretation and next boundary

This test measures the net effect of the removed input representation. It removes
both noisy matches and genuine causal full-prefix information in the literal
register, and it shortens prompts. Even a large gain would not isolate distraction
as the cause. Both arms still receive privileged correct previous state; neither
is autonomous recurrent memory. The cohort has been exposed repeatedly and the
intervention was motivated by post-result diagnosis.

Do not resolve the two reviewed preference-versus-action label disagreements by
editing targets. Do not train from reviewer answers in this experiment. If the
rule passes, use the stronger observation control to test an explicitly defined
memory mechanism later, with predicted-state rollouts and fresh evaluation under
shift. If it fails, report which tradeoffs remain before selecting another
hypothesis. No recurrence, connectome, RL or biological-learning advantage follows
from either outcome.
