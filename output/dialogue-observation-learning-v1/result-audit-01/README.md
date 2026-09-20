# Independent saved-results arithmetic

Prepared while the twelve-fit scientific attempt is running. No actual task
predictions or evaluator rows have been decoded by this audit yet. Nine small
synthetic tests passed; their receipts and logs are retained under `preflight-01`.
These checks establish implementation behavior, not a scientific result.

`audit.py` uses only Python's standard library and NumPy. It authenticates the
complete run, frozen sources, independent supervisor terminal and production
report before loading predictions. It independently recomputes all/seen/unseen
panel and stratum scores for every fit, both literal accuracy controls,
equal-seed arm means and the seven primary conditions. Accuracy margins use
exact fractions; NLL uses saved float64-promoted logs directly; Brier sums over
candidates. No probability repair or excluded endpoint is permitted.

Detailed training/configuration validation, source data joins and literal-rule
semantics are inherited from the authenticated production report. Service/type
tables and descriptive factorial contrasts are outside this independent audit's
scope. It neither replays training nor loads checkpoints.

The audit requires externally supplied hashes for the run completion, frozen
plan, supervisor launch/terminal and report receipt/summary. Run it only after
all twelve fits and the production report finish, into a new exclusive output
directory. Its prospective limits are 300 seconds, 4 GiB RSS and 256 MiB output.
Any disagreement or limit failure preserves a failure receipt.

Source SHA-256: `054199c8a60b2b9c3ad47fda8fd61faef87a9e2020b683593f00dc67aca3d548`.
Test SHA-256: `5337fd47f792cb5cc029292a1191f8d368b1bc7294f0e419d65b8697703d776e`.

The synthetic preflight briefly overlapped the live scientific worker on one
CPU thread. The timing-context supplement records that fact retrospectively;
it is not a standalone latency benchmark. Raw scientific inputs remain local.
