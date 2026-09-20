# Inspect every delayed SYSTEM-only proxy

Follow-up fixed after the complete support diagnostic, 20 September 2026.
The diagnostic found 38 proxy rows across 37 dialogues and 12 services.
Inspect **all 38**, ordered by dialogue ID, USER turn and query ID; no example
selection or model fitting. The raw literal-history result remains unchanged.

Use the hash-authenticated endpoint metadata, full public TRAIN prefixes and
supplied TRAIN catalog. No future turns, system-act annotations, DEV dialogue
file, official TEST content or model prediction enters the inspection. Preserve
the four-exchange suffix boundary and the previous annotated endpoint.

This is a **label-aware source inspection**, not a blinded model evaluation.
The reviewer sees the official target to check what the lexical proxy actually
captures. For each case, record whether visible older context is needed to
interpret a delayed reference, whether a public recent exchange already
expresses the target semantically, or whether the case is ambiguous. Separate
cross-service transfer, incidental numerical matches, referent ambiguity and
annotation gaps. Multiple explanations can coexist. Do not infer a population
accuracy, causal effect or trained-model gain from these judgments.

Store the complete review packet locally with hashes. Publish case IDs,
turn-index references and concise paraphrased assessments, not a raw dialogue
dump. Aggregate the complete reviewed set and state the next research decision.
If delayed references are credible, they establish a small candidate diagnostic
set, not automatic training admission or enough evidence for a novel recurrent
architecture. Do not alter labels or any earlier experiment decision.
