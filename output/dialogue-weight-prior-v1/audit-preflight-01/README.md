# Independent weight-prior audit preflight

Thirteen synthetic tests passed in 0.16 seconds and Ruff passed on the first recorded attempt. Exact stdout/stderr, command arguments, process exit codes, elapsed times, and before/after source hashes are retained in attempt-01.json and its logs. There were no test or lint failures in this preflight.

The auditor depends explicitly on the frozen independent commitment auditor for authentication and primary metric helpers. The new correction independently adds the analytic FIT-count log-odds shift to the public previous candidate before logaddexp normalization. Production code imports no diagnostic producer, model, encoder or trainer. Producer arithmetic is imported only by an artificial compatibility test.

Coverage includes NONE/assigned priors, nonzero NONE index, two-candidate support, candidate permutations, padding, extreme finite log probabilities, unchanged conditional alternatives, analytic odds, float64-versus-float32 weight reporting, raw-normalized loss shifts, label independence, invalid inputs, exclusive failure preservation, and all-eighteen artificial producer-to-independent-checker output agreement. Real saved predictions remain unopened; this receipt is not a scientific result.

A separate source-only peer review found no material blocker at the final source hash. It did not rerun tests or access predictions.
