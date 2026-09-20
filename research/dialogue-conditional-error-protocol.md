# Conditional decisions: posthoc error diagnosis

The nine-fit conditional-observation study is complete and failed its primary
continuation rule. This analysis was chosen **after observing that failure**.
It cannot change the original outcome, select a seed, remove ambiguous labels,
reweight evaluation, repair probabilities or authorize another memory fit.

Use only the completed run pinned by SHA-256
`df3c172bae7163292b54cdd9a3b1d6e3bb5a07acb49b62be4f0c0dfc68efbe75`
and original analysis summary
`fb065bcc05550734f7fbe8a24def71b4b69ae5288c06dabbe1caddf39fb50e20`.
Authenticate the complete execution through the frozen saved-output reader.
Read its admitted train/dev metadata and all nine final prediction arrays.
No model, optimizer, encoder, engine, new labels or official test access.

For every original seen/unseen group and every additional exhaustive
seen/unseen by transition by current-value cell, report correct choices,
changed-row prior choices, and other wrong choices. These partition predictions.
On retained rows, choosing the prior is correct, so it is not a separate error.
Report target/prior/strongest-other probabilities, target NLL, and the mean
target-minus-prior log-probability margin on changed rows only. Classify wrong
alternatives as reserved NONE, DONTCARE, schema-Boolean TRUE/FALSE, or other.
Literal ontology `None` is not reserved NOT_MENTIONED. Preserve exact top-1 ties
and the existing first-index argmax convention. Do not floor probabilities.

Report admitted TRAIN support by transition/current-value and previous/current
value class, with row, dialogue and query denominators. The old three-stratum
weights do not imply balance among every value class. Keep unique-row counts
separate from the nine fits' prediction-event counts. Empty conditional means
are undefined, never zero.

Decompose each paired primary NLL difference by the five disjoint current-value
classes, dividing each group's summed difference by the **full primary row
count**. Contributions must reconstruct every published seed difference and
their mean to absolute tolerance 1e-12. Report the group's conditional mean
separately. Do not report percentage shares of a small net difference formed by
opposing contributions. Absent groups contribute zero to the sum but have no
conditional mean.

Publish source and synthetic checks before one saved-output analysis in an
exclusive directory. Preserve its source and input hashes and any failure.
This is descriptive diagnosis, not a new benchmark or primary result. Prior
choice errors do not prove recurrent inertia: the scorer has no recurrent carry
mechanism. Low target probability cannot establish missing information or a
causal feature defect without a separate intervention. Manual wording checks
and paper-based hypotheses must remain distinct from measured quantities.
