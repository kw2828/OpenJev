# Fixed-memory controller comparison

**Prospective only. No training, model, environment or random-stream calls were made for this design.** The original pilot failed its frozen continuation rule, passing 1/6 checks. That result stays unchanged. This is a separate fresh-deck evaluation of the same final checkpoints, motivated by a saved-score numerical diagnostic, not an architectural rescue or a reopened training run.

The [completed saved-score diagnostic](picker-diagnostic-01.json), SHA256 `7171d8792a6a0a98c78e46aba37637addb4fd5d8cae5bc6b16313c94ba2a1c20`, authenticated the original 1,280 episode payloads and compared choices at actual saved public prefixes. Its normalization changed neither trajectories nor memory state. The parent reports that it changed 4,026/9,984 first-card choices, with no second-card changes; this does not establish a return improvement. Only the diagnostic's compact metadata was inspected for this note. Its `1e-7` threshold described score gaps and was not an action tie tolerance. The new `1e-12` tolerance below is an explicit, prospectively fixed policy rule.

## One three-picker evaluation

Use every **18 final checkpoints**, preserving all six families and three paired fits, plus the **exact-public-table and last-32-selected-reveals references**. Evaluate each of these 20 controllers with all three pickers on the **same 64 fresh native ConcentrationHard decks**. Coverage is exactly **3,840 episodes**, at most **399,360 native actions** at the unchanged 104-action limit. The maximum learned predict/write counts are each 359,424; the references account for up to 39,936 actions. No fourth picker, horizon extension, retraining, new memory state, reward change, or checkpoint selection.

Use a new frozen reset-seed list, disjoint from the original training, development, evaluation and engineering lists. Do not allocate it as part of this note. The reset seed identifies the paired deck across all 60 controller/picker combinations; there are 64 independent layout draws, not 3,840 independent layouts. Retain all decks, including accidental duplicate layouts, and report their count. Freeze execution order before launch, balancing picker order across deck index without outcome-dependent scheduling. Native trajectories may diverge after different actions; they must not be described as identical later observations.

### A. Original picker

Call the exact frozen original picker with its original float32-softmax values converted to float64. Keep its identical public unseen/visible overrides, legal-action restriction, computed-score argmax, and lexicographic order. No renormalization or tolerance. This fresh-seed row is a paired reference, not a replacement for the original pilot's score.

### B. Normalization plus tolerant lexicographic ties

Validate finite nonnegative rank probabilities and strictly positive row sums. Convert to float64 and divide **each raw row** by its float64 row sum. Then apply the unchanged public overrides: unseen positions are uniform and currently visible positions have their public one-hot rank. This ordering matches the saved diagnostic. Do not claim unit mass is exactly representable after finite arithmetic.

For a first-card decision, enumerate eligible unordered pairs `(i,j)`, `i<j`, in the original lexicographic order. Compute `s(i,j)=sum_r p(i,r)*p(j,r)` in fixed float64 arithmetic. Let `T` contain every pair with `s_max-s(i,j) <= 1e-12`, inclusively. Choose the lexicographically first member and issue its first endpoint. For a second-card decision, apply the same absolute tie band to the eligible cards' probabilities of the currently open public rank, then choose the smallest index. The source, reduction order and exact threshold comparison are frozen.

**A versus B tests the combined numerical-policy change.** It cannot separately attribute improvement to row normalization versus the new tie band. The requested three arms are sufficient to separate that bundle from the exploration preference below; a fourth numerical ablation is unnecessary for that question.

### C. Same normalized scores, information-seeking first tie

Use exactly B's probabilities, scores and pair set `T`. Gather unseen endpoints of every pair in `T`. If any exist, issue the smallest-index unseen endpoint; otherwise issue B's first endpoint. Second-card decisions are exactly B's rule. No bonus changes the expected-match score, and an unseen card outside the fixed tie set is ineligible for this preference.

The endpoint rule is intentional: even a unique best unordered pair has two first-card orientations with the same match score. C may choose its higher-index unseen endpoint first. Without this rule, upper-triangle indexing can silently prevent the intended exploration. The only information used is the shared public `seen` mask, current observation, matched mask and pending index. There is no additional historical rank cache, exact teacher access for learned models, unseen-card label, or lookahead simulation. This is a simple decision heuristic, not optimal information gain.

## Evidence and interpretation fixed in advance

Primary controller contrast is **C minus B**; **B minus A** is the separate numerical contrast. Report mean native return, completed pairs, success and action count for every family/pair/picker and both references. Aggregate each fit over all 64 decks, then families over all three fits. Show every paired fit difference and deck-level paired differences; neither first-card decisions nor repeated fits create additional independent layout samples. No best-family or best-seed substitution.

A concrete proposed claim-support rule for a *practical exploration benefit* is C-minus-B mean return at least **0.03 across all 18 learned fits**, nonnegative in every family mean, and strictly positive in at least two of each family's three fits. Report reference contrasts separately. Failure of this rule means the general exploration-benefit hypothesis is unsupported at this budget, even if one model benefits. This criterion is new and must be frozen before fresh inputs; it does not alter or reevaluate the original 1/6 architecture gate. Do not apply a new favorable architecture gate after seeing these results.

Also report raw row-mass drift, first/second tie-set sizes, unseen availability inside the first tie set, fraction choosing unseen, public positions discovered, and revisits. At each actual C prefix, record whether its chosen action differs from B's choice computed from the same saved scores. These are bookkeeping diagnostics, not extra counterfactual trajectories. If C improves while B does not, that supports an exploration preference, not merely a precision fix. If all families improve similarly, the evidence points to the shared controller. An architectural comparison needs separately prespecified fresh validation and cannot inherit novelty from this intervention.

## Execution boundary and complete costs

Before the first reset, authenticate the original protocol, completed training and independent review, all 18 checkpoint/configuration/member identities, and the exact two reference definitions. Checkpoint tensors remain unchanged before and after the entire comparison. Reset only episode-local model state for each episode. No development re-selection or updated optimizer state is allowed.

Add pure tests before launch for exact ties, gaps just below/equal/above `1e-12`, row scaling, public overrides, a higher-index unseen endpoint, no unseen tied endpoint, and identical B/C second-card choices. Preserve original A parity on saved inputs. Run the one frozen evaluation once; incomplete episodes or a cap failure remain failures, with no silent retry or selective denominator.

Save the native action/observation/reward/terminal record plus raw and picker probabilities so choices can be reconstructed without neural reruns. Two float64 `[52,13]` probability arrays per action have an upper bound of **4,319,477,760 uncompressed bytes** across this design, before other traces. Include this storage and serialization cost in sizing. Record restore, public bookkeeping, normalization/tie selection, inference/write, native stepping, storage/hash and whole-run times separately. Reusing weights costs zero new training steps; it does not make deployment free or compute-matched across families. Set a single hard wall/storage cap from the completed original phase's measured costs before launch, with no outcome-driven extension. The new saved audit must distinguish public-choice replay from any separately authorized native replay.
