# Qualifying the candidate-evidence bottleneck

Status: pre-replay specification. This inspects six existing trained models from the completed [dialogue-copy study](dialogue-copy-results.md). It fits no weights and changes neither that study's predictions nor its failed continuation decision. Freeze this protocol, the diagnostic source/tests, existing scientific sources, input receipts, checkpoints, and runtime before replay.

The [saved-error analysis](../output/dialogue-copy-error-diagnostic-v1/README.md) found both missed replacements and broader interpretation failures. In particular, an explicit current USER mention supports the new gold on some unseen revisions where the model keeps the old gold after a correct previous prediction. However, old final predictions do not show whether the writer proposed the correct new value or whether the replacement gate suppressed it. Stale output alone does not identify internal belief inertia.

## Fixed scope

Replay every development query for `scalar` and `selective`, with seeds 4101, 4102, and 4103: six checkpoints, 62,329 scored queries each. Do not select successful seeds or difficult examples. Use the original CPU float32 forward path, four threads, batch order/size, frozen packet, lexical cache, and checkpoint bytes. No official test contents, fresh encoder calls, optimizer, model selection, or training are allowed. The original development split has already been inspected repeatedly.

The original plan SHA-256 is `609a0758f2a4ae4722052bb27740c74b3d1bdac396d8134eb39f5118f1429a2b`; completion SHA-256 is `8317abad5ef8235f31f5381591944ad152c42fddb98743fa64fd1f5cd8c3f300`. Retain the same original 13 scientific source identities, plus the new qualification implementation, tests, and this document. The successful feature and lexical receipt hashes remain `e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308` and `2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54`.

A subclass may capture detached tensors around the unchanged `_advance` call. It must not change the parent forward path, actor inputs, parameter values, or normalization convention. Load only hash-verified weights using `weights_only=True` and strict key/shape checks. Record and restore constructor RNG state; constructor initialization is discarded when loading the complete checkpoint.

Match every replayed row to its original dialogue/time/question/candidate identities and labels, and compare the final candidate probabilities with the saved predictions. Require maximum absolute difference at most `1e-6`, record bitwise equality separately, and stop on mismatch. A successful source/shape check alone does not qualify a replay. Preserve failures and partial progress; do not silently replace or retry a replay. Use a 300-second wall-time cap, checked between batches, and retain the timeout receipt if exceeded. This is an execution bound, not a search budget for extra variants.

## Captured factors and arithmetic interventions

For each scored row, retain the model's prior candidate belief `b`, current writer probabilities `w`, released mass `m`, and actual result. Individual factors and probabilities stay in ignored local run files. The actor receives only the original public text features and supplied schema; gold labels and transition categories enter diagnostic partitions only.

Compute two **one-step counterfactuals from those fixed factors**, without new model calls:

- Forced write: return `w`.
- Normalized product: set `pi=(1-m)*b+m*uniform_valid`, then normalize `pi*w` over valid candidates.

These intervene in one update along the original model's trajectory. They do not propagate changed beliefs to later turns and are not trained alternatives. Their repair/break counts must never be presented as achieved rollout accuracy or the measured performance of a new architecture. The scalar and selective operators use `m` as released mass; its role in the counterfactual product is support renewal, a different operation. No equal-release or exact-Bayesian claim is allowed. Add no undocumented probability floor.

Numerical convention, fixed before replay: retain the actual captured float32 factors, then perform counterfactual arithmetic in float64 without clipping `m` or renormalizing `b` or `w`. Permit only normalization error at most `2e-6` for each saved distribution and `0 <= m <= 1+2e-6`; a tiny overshoot can arise from the parent's float32 log-space arithmetic. Record the number and maximum size of `m>1` overshoots and the maximum normalization errors per fit. Require the resulting `pi` to be finite and nonnegative and the product normalizer strictly positive. Larger deviations fail the diagnostic. This tolerance is numerical validation, not a learned threshold or a probability correction.

## Required diagnostics

Retain every fit and report all/seen/unseen panels with unmentioned retention, assigned retention, first assignment, revision, and clear strata, including absent support. Distinguish public-time adjacency from an intervening unscored turn.

For revisions, partition wrong outputs into stale previous-gold values and other errors; separate cases with a correct versus incorrect previous scored prediction. Within those groups, record whether the writer's argmax is the current gold, previous gold, or another value. Further partition by unique current USER-gold match and preceding SYSTEM-gold match. Matches identify strings, not verified intent.

For retained labels, distinguish newly wrong predictions after a correct prior prediction from already-wrong predictions. Count counterfactual repairs of wrong results and damage to correct results, with their denominators, rather than reporting repairs alone. Include the assigned-boolean and DONTCARE interpretation deficits from the saved-error diagnostic as context; this qualification does not test whether those deficits can be fixed by gating.

Report fixed quantiles `[0, .25, .5, .75, 1]` of relevant old-belief concentration, writer confidence, and released mass. These are descriptions, not thresholds to optimize a routing policy. Keep integer per-fit counts; fractional aggregate counts arise only from equally averaging the same three seeds.

## How this changes the next action

The main question is whether a correct writer proposal is being suppressed on the explicit USER-supported, previously correct, adjacent unseen revision errors. A majority of wrong writer proposals points toward improving observation interpretation. A majority of correct writer proposals supports testing the missing competitor-to-gate input path, while still requiring a matched learned comparison. Report absolute counts so a tiny subgroup cannot be mistaken for a general solution.

Neither dominance finding guarantees that a new gate can recognize the relevant cases from public inputs. Counterfactual product repairs also need their corresponding break counts, especially on retained states. The [provisional design](dialogue-evidence-design.md) and [source review](dialogue-evidence-source-review.md) remain conditional options. Ordinary learned belief filtering, copying, and product updates are established. No training, new architecture claim, calibration claim, or ICLR-readiness decision follows from this diagnostic alone.
