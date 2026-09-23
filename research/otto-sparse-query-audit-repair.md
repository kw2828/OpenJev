# Sparse-query saved-audit correction

The scientific process completed all 384 paths under its original allocation.
The first saved-record audit then failed at its scalar cost check: it applied
`isfinite`-style scalar validation to every key ending in `seconds`, including
the nested `operation_seconds` dictionary. The producer's declared schema has
always contained that dictionary. This is a checker defect, not evidence of a
negative physical time. The original failed audit and all frozen sources remain
unchanged.

Before reading scientific summaries, prepare one separate corrected verifier.
It must validate the exact scalar cost fields as finite, nonnegative numbers,
retain independent checks on the nested operation records, and retain every
original episode, chronology, query, action, threshold, cost and condition
comparison. Bind the corrected source and tests, the original scientific plan,
producer receipt and successful original terminal, and the failed audit receipt
and terminal in a new repair plan. Qualify the correction on fabricated records
and review the source before running it.

The corrected saved audit gets one separate 120-second supervisor allocation,
2 GiB RSS and 128 MiB output. This adds verification cost; it does not extend the
scientific allocation. There are no new model, native environment, posterior
filter, sampler, array-decode or optimizer calls. There is no change to seeds,
arms, threshold, data, action traces, costs, sixteen primary conditions or
fifty-four diagnostics. Any remaining mismatch must be preserved and reported.

Publish both audit attempts. Successful corrected agreement may verify the
unchanged saved results, but must never be described as success by the original
frozen checker. Charts and reports must link this correction and both attempts.
No result is promoted on the basis of the failed original audit.
