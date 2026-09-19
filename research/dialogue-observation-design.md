# A conditional observation-representation control

Source-only proposal, 2026-09-19. No V2 partial results, new corpus examples, encoder calls or fits were inspected or executed. This is conditional on the corrected replication, not an automatic next study.

**Subsequent result:** the [completed V2 replication](dialogue-copy-v2-results.md) retains the deficits. Its saved-output diagnostic finds that all 524 unseen assigned-boolean targets are TRUE; FALSE support is zero. Any follow-up using this panel must describe TRUE/schema transfer and DONTCARE recognition, not general polarity or a demonstrated negation deficit. This proposal has not been run.

The historical selective model's poor unseen boolean and DONTCARE predictions motivate an interpretation check. They do not establish missing semantic information, and the old scalar comparison is compromised by the subsequently discovered state-normalization defect. Any new comparison must use the normalized V2 implementation and its completed results.

## What the current representation actually contains

[The encoder](../scripts/study_dialogue_memory.py) uses frozen `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Each observation is the immediately preceding SYSTEM utterance followed by the current USER utterance, with explicit `System:` and `User:` labels. It tokenizes without truncation, encodes nonoverlapping 254-content-token chunks, mean-pools each chunk including special tokens, combines chunks weighted by content-token count, and L2-normalizes to 384 dimensions. Cross-chunk attention is absent, but no source review establishes that chunking caused observed errors.

[Schema preparation](../src/openjev/research/dialogue_state_data.py) already puts service and slot descriptions in every query and candidate string. Candidate strings also include the value; NONE and DONTCARE include explanatory text. Thus neither roles nor schema semantics are literally absent. The narrower limitation is that the turn is pooled **before** interacting with the requested schema or candidate. [The learner](../src/openjev/research/dialogue_copy_memory.py) later combines projected turn/query/candidate vectors, their elementwise products, ten public lexical features, and belief features. It never accesses token-level turn/schema alignments. Its boolean regex cues are noisy public observations, not semantic polarity annotations.

## Relevant primary evidence

- **SGD, Rastogi et al., AAAI 2020, Section 5:** separate schema embeddings describe slots and categorical values; BERT encodes the preceding SYSTEM and current USER, and prediction combines the two. Its status classifier distinguishes none, dontcare and active. This confirms a conventional schema-conditioned formulation, not that our pooled MiniLM is sufficient. [Primary paper](https://ojs.aaai.org/index.php/AAAI/article/download/6394/6250).
- **SUMBT, Lee et al., ACL 2019, Sections 2.1-2.3:** the slot representation queries contextual utterance-token vectors before the recurrent tracker. The utterance BERT is fine-tuned, while the slot/value encoder is fixed. This is direct precedent for conditioning evidence extraction before recurrent updating; its results do not transfer to our frozen encoder or restricted task. [Primary paper](https://aclanthology.org/P19-1546.pdf).
- **TripPy, Heck et al., SIGDIAL 2020, Sections 3.2-3.7:** user spans, system-informed values and prior state have distinct roles; boolean and dontcare cases use classification rather than literal span copying. It additionally uses dialogue history and system-inform information. Those richer inputs and slot-specific heads are not equivalent to our public-text-only, schema-shared setup. [Primary paper](https://aclanthology.org/2020.sigdial-1.4.pdf).

## One bounded control

Compare **independent encoding** with **candidate-conditioned joint encoding**, using the same frozen MiniLM weights, pooling, normalization and all-token chunking rule. Joint text contains the existing service/slot/candidate description followed by the same labeled SYSTEM+USER pair. Its 384-dimensional vector replaces only the broadcast turn vector for that candidate. Keep the original query/candidate vectors, ten lexical inputs, projections, heads and V2 normalization. This is joint encoding with a frozen sentence encoder, not a pretrained entailment classifier or a new copying mechanism.

Use a two-by-two comparison: old versus joint representation, each with **V2 scalar** and **readout**. The readout still receives the existing deterministic lexical history; it has no learned belief recurrence. Match initial head tensors, training examples, batches, loss, updates and final-only evaluation across representations. Candidate-specific evidence changes tensor indexing, not the learned parameter count. Retain literal carry and the completed V2 controls as references. No encoder fine-tuning, new negation rules, gold-selected candidates or operator changes belong in this control.

Every supplied candidate, including NONE/DONTCARE, is encoded independently with the same template. Candidate permutations must only permute outputs; unrelated queries must have no effect; changing future dialogue text must leave earlier evidence unchanged. Query-conditioned public-prefix replay is allowed, but scored-query eligibility and frame annotations cannot select actor updates or encoding content. Joint chunking may place schema text outside later chunks; retain that limitation and report affected counts rather than silently truncating or changing the rule after results.

**Falsifiable hypothesis:** exposing schema/value text to the frozen encoder before pooling improves unseen assigned-boolean and DONTCARE recognition under *both* matched readout and scalar heads, without sacrificing overall unseen macro accuracy. Freeze denominators, paired comparisons and a practical minimum gain before any encoding/fitting. A gain under both heads supports a usable representation limitation under this recipe, not information-theoretic loss or a superior memory operator. A scalar-only gain indicates an interaction. No gain closes this particular frozen-joint-encoding control; it does not prove that text interpretation is solved or justify another gate automatically.

## Cost and decision limits

Joint encoding scales with public turn-query-candidate combinations instead of unique turn texts. Before authorizing fits, measure its bounded engineering capacity and account for all actual token/chunk counts, encoder time, peak memory, cache bytes, preprocessing and training/evaluation time. Report both cache-amortized head latency and uncached request cost; a shared offline cache is not free inference. Twelve fits would be the full four-arm, three-seed comparison, with the old recipe's update count per fit, not an authorization here. The development set is already exposed, so any result remains a development finding. If V2 resolves the relevant deficits, or the representation cost is impractical, do not launch this screen.

## Source identity

| Inspected source | SHA-256 |
|---|---|
| `scripts/study_dialogue_memory.py` | `51a46d35c3d9dd2857bd3ded5fc289652c7e18f05b85b8f82dd245ade7626e2c` |
| `src/openjev/research/dialogue_state_data.py` | `7749e97e64575ae02c9fc843573e93e82d5e9cbb301410e47881e18190f0c102` |
| `src/openjev/research/dialogue_copy_memory.py` | `182c71a944c6d787533bf129bca1631ff3b366b8559e3db7567e7fd9b19a8717` |
| `src/openjev/research/dialogue_copy_memory_v2.py` | `3fc1e84e5fe9da0076e8d83e7d5c67e61607d67b49ce2b0357a81e2206d3714b` |
