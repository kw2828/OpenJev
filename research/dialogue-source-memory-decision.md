# Conditional decision on source-specific dialogue memory

Source review, 20 September 2026. This note is not a protocol, admission, or
authorization to fit a model. No new predictions, data inspection or experiment
was performed. **Cancel this architecture branch if the six-fit row-uniform
[objective study](dialogue-objective-protocol.md) resolves the prescribed
changed/retained tradeoff.** In that case, prefer the simpler objective and
establish autonomous scalar tracking before adding memory structure.

## What motivates the question

[Token alignment](dialogue-token-alignment-scientific-results.md) improves
changed-state accuracy from 78.60% to 83.45% against token mean, but increases
retained error from 6.44% to 7.52%. Its additional retained errors concentrate
on slots that should remain unmentioned: 275 extra prediction errors across
the three seeds, versus 42 fewer errors on assigned retentions. These are
repeated predictions on the same examples, not independent cases.

[Fixed training-weight correction](dialogue-weight-prior-results.md) reduces
aligned retention error to 4.82% while lowering changed accuracy to 75.89%.
Corrected flat remains strongest on pooled overall accuracy at 94.66%.
Consequently, the evidence establishes a decision tradeoff, not a need for a
new recurrent operator. The earlier [conditional failure](dialogue-conditional-error-results.md),
including all nine fits missing all 31 TRUE updates despite correct supplied
previous values, also warns against attributing semantic mistakes to memory.

Source confusion is an **unproven hypothesis**: a candidate can be relevant to
a SYSTEM proposal without being adopted as the user's preference. These
aggregate results do not show that SYSTEM mentions caused the false updates.
Speaker roles are already available. The [token preparation](../scripts/prepare_dialogue_tokens.py)
uses `System: ...\nUser: ...`, and the [lexical features](../src/openjev/research/dialogue_copy_features.py)
separate USER and SYSTEM matches. SYSTEM mentions already cannot write the
literal register. This proposal therefore concerns use and persistence of
existing source information, not correcting an absence of role labels.

## One conditional mechanism and its controls

Keep a question-specific proposal state separate from the committed candidate
belief. Public SYSTEM evidence updates the proposal, retaining candidate
identity and source turn. USER evidence can carry the committed belief,
write a directly expressed value, or refer to a proposal. A SYSTEM mention
alone does not commit a user preference. Interpretation is learned from public
text; no handwritten yes/no rule or annotated system act enters the actor.

Any autonomous follow-up must initialize and maintain **model-generated**
proposal and belief states. It cannot supply gold previous values or reset
after mistakes. Current labels supervise losses only. Process public turns
independently of annotation availability, retain candidate/query independence,
and charge state, extraction and full decision cost. Existing conditional
caches are not claimed to be a ready autonomous-stream fixture.

The main control is a competitive scalar ledger with the same role-aware
features, scorer, objective and last two complete exchanges. Also require an
unrestricted two-register model with matched public information and comparable
state/capacity, so an improvement cannot be credited merely to extra storage.
The proposed intervention is the restriction of writes by source and the
separation of proposals from commitments, not a constant prior adjustment.
It makes no universal expressivity claim against the learned scalar control.

The hypothesis is falsified if short history or the unrestricted registers
explain the improvement, or if fewer false assignments simply cost more true
changes. Useful evidence would preserve changed decisions and proper scores
while reducing unmentioned false assignments during autonomous tracking.
There is no new numerical gate here, and source confusion would still require
direct diagnostic evidence rather than inference from aggregate gains.

The [robotics control](reacher-two-observation-control.md) motivates this
discipline: persistent GRU's earlier 31.86%/25.99% improvement against a
one-observation reset narrowed to 4.44%/2.94% against two observations plus
actions, failing the stronger rule. It does not prove short-history
equivalence or transfer that mechanism to dialogue.

## Established overlap and limits

[TripPy](https://aclanthology.org/2020.sigdial-1.4.pdf), sections 3.4-3.8,
already separates system-inform memory from dialogue state and copies an
informed value when the user refers to it. Its structured inform information
is not available as an extra input here. [SOM-DST](https://aclanthology.org/2020.acl-main.53.pdf),
section 3, already separates carry/delete/update decisions from value
generation. Neither source supports calling another operation head novel.
[Gated Delta Networks](https://arxiv.org/html/2412.06464v1), section 3.1,
combine gated forgetting with associative delta updates; that algebra does
not itself distinguish a proposal from an adopted preference.

This revisits the locally proposed source-memory idea in the
[earlier review](dialogue-evidence-source-review.md), rather than introducing
a new discovery. All cited local dialogue outcomes are exposed development
evidence. No novelty, calibrated confidence, autonomous memory efficacy,
world-model capability, or revision of any failed rule is claimed.
