# What would justify another learned-memory experiment?

Historical evidence review, 20 September 2026. The lexical-input ablation is
still running; no predictions from that run were inspected for this note.
This is a conditional research decision, not a frozen protocol or an admitted
training run.

The subsequent [primary-source prior-art review](dialogue-commitment-prior-art.md)
finds direct precedents in TripPy and SOM-DST. Separate proposal memory, carry
operations and update-before-value decisions are not novel by themselves.

The current observation comparisons supply the correct previous value. They
test interpretation and retention of known state, so an improvement cannot
establish autonomous memory. Adding a recurrent adapter or biological wiring
to this interface would leave that identification problem unresolved.

Earlier [corrected candidate memory](dialogue-copy-v2-results.md) did show a
useful learned-state effect within its recipe: scalar memory reached 72.58%
unseen-service macro accuracy against the tested readout's 65.77%. But selective
memory reached only 72.89%, lost to scalar on seen services and failed its
continuation rule. The [training-objective comparison](dialogue-objective-results.md)
also changed the revision/retention tradeoff without changing architecture.
Neither result justifies another topology variant by itself.

The [Qwen retention diagnosis](dialogue-qwen-retention-results.md) motivates one
specific hypothesis: distinguish an unresolved proposal from a committed user
preference. Its error-conditioned sample also exposed ambiguity between a
persistent preference and a selected option. These observations do not establish
error prevalence or justify changing reference labels.

If a clear history-dependent gap remains after the current ablation, compare
a small learned proposal/commitment accumulator with a strong explicit ledger.
The ledger should retain committed value, latest proposal, its source and age,
plus the same short public history. Matched controls would be the same learned
accumulator with carried hidden state reset and an ordinary small GRU. Share
the observation features, candidate interface, supervision and objective;
count ingestion, retained storage and decision computation for every method.
Role features alone are not a new mechanism because earlier models already
received them.

An unrestricted two-register control must also match storage and observation
inputs to distinguish the proposed write rule from extra capacity. The explicit
ledger's proposal extractor must use the same public text and paid observation
computation, not annotated system acts. Carry and clear are distinct operations;
an unmentioned-state candidate must not silently become a no-update instruction.

Every method must maintain its own predicted state over the complete public
stream, including unscored turns. Do not inject correct previous values or reset
to gold at scored endpoints. Establish an unambiguous target and enough delayed
revisions, corrections and anti-updates before fitting. The current repeatedly
exposed short streams and rare transition types cannot support a general claim.

The learned mechanism is falsified if carried state cannot improve both
revision and retention behavior over the ledger and reset controls, or if the
gain comes only from richer observations or the extra explicit register. Stop
this line on the current task if remaining errors mainly concern present-text
interpretation or incompatible state definitions. A later fresh protocol must
specify the actual cohort, resource limits, effect margins and shift test before
execution. No thresholds are selected here.

Only after ordinary recurrence earns a repeatable contribution should connectome
constraints enter a comparison with matched random sparse and dense controls.
A proposal/commitment accumulator is a conventional mechanism hypothesis, not
an established OpenJev innovation or an ICLR contribution.
