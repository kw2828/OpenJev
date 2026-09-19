# Proposal-memory implementation review

Status: no material implementation or arithmetic blocker found in the components listed below. This is a pre-execution source review and synthetic orchestration check, not an engineering outcome or scientific qualification. The independent native replay auditor is being finalized separately and is outside this completed review.

## Exact reviewed sources

| Repository-relative path | SHA-256 |
|---|---|
| `src/openjev/research/reacher_cem_proposal_memory.py` | `65935bf14870cae33a58d5218b78aa089f6efe55bd56f8a57721e7da0e2e5106` |
| `tests/test_reacher_cem_proposal_memory.py` | `597cbb41da1f860490980efa67669078499ffb6bc47b4c57b3b4e5a781d0bee9` |
| `src/openjev/research/reacher_proposal_memory_rollout.py` | `cee7edbc46f9c162528d36705ca7e4894b898238fb8679164488d10422d4a0a9` |
| `tests/test_reacher_proposal_memory_rollout.py` | `1a903d0340f1a885d45e5743f2342e7003e508b799e462b56859372a79c3865f` |
| `output/reacher-proposal-memory-engineering-v1/run_engineering.py` | `29c6ce379ebe1a492923d562936c6dd0ffc409534f215822d9824993e1cf4779` |
| `output/reacher-proposal-memory-engineering-v1/protocol.json` | `7ffea079cfaea91eb71ce091db54123fa8d0666966325488a729f61f08fe2711` |
| `output/reacher-proposal-memory-engineering-v1/report_results.py` | `859d7dcf7b6d3ade1672a16caf02997e283e2345c5ac16a61a299b771a1c5e69` |

The driver's change since the earlier review is inclusion of the report source in its recorded and copied source set. Original frozen sources were not edited by this reviewer.

## Causality and search semantics

The component shifts a committed selected sequence by one actually issued action, hold-pads only the remaining far tail, and takes explicit float64 column means within current three-action blocks. Only initial random slots 7:64 are translated. The seven anchors, original scales, subsequent CEM innovations, four evaluated banks, paid final mean and earliest global best remain governed by the unchanged search kernel. A zero center returns the identical input object. Inactive terminal blocks have zero centers.

The memory receives only the current observed public target, selected sequence and actual command acknowledgment. Target changes reset the next proposal center; hidden gain changes and future target schedules do not. The rollout passes transformed inputs to the policy, confirms the native packet and issued command, completes the public observer update, then commits proposal memory. Failed preparation/commit or observation cannot silently advance a successful prefix. Snapshots and trace copies do not alias the cache.

## Verification performed

- Component author: **35 pure synthetic tests passed in 0.09s; Ruff clean**, reported by the author. This reviewer read the final source/tests and checked their hashes, but did not rerun those tests.
- Reviewer-authored rollout tests: **22 passed in 0.92s; Ruff clean**, executed with `PYTHONPATH=src .venv-robotics/bin/python -m pytest -q tests/test_reacher_proposal_memory_rollout.py`. All environments, policies, search results and memory operations in these orchestration tests are handwritten fakes. They check all nine role/mode cells; transformed and original input identities; exact native acknowledgment before observation/commit; target-event eligibility; ragged failure prefixes; original exception preservation through secondary write failures; late completion demotion; role admission and exclusive output.

No model, simulator or RNG call was made for this review. No completed or partial proposal-memory run artifacts were read, and no report was rendered.

## Enclosing driver and report

The reviewed protocol contains all three roles crossed with all three proposal modes. The driver authenticates the prior completion, row completions, audits, source hashes, exact reused case fields and all original innovation bytes. Its cold-mode check compares every saved compact decision array, including candidates and scores, and the entire native episode against the corresponding prior row. All nine rows must finish before the nine native audits and interpretation. Exclusive output, cooperative execution/audit caps, late completion demotion and original-error preservation are retained.

The derived scheduled work is **1,800 real transitions, 5,377,536 candidate transitions and 1,800 separately charged selected advances**, with no identification updates. The 600-second execution clock includes setup/source snapshots after attempt allocation; preliminary source/parent authentication occurs before that clock. The separate 600-second audit clock includes cold parity and exit checks. These are cooperative checks, not an external preemptive process timeout.

The report uses action intervals **full [0,200), pre [0,80), post [80,200), move [100,150)**. It negates native rewards to obtain cost, divides interval means by the corresponding 200/80/120/50 actions, and preserves all nine rows. Its six checks are exactly three roles times two controls: positive baseline cost and `shift_plan_cost <= 0.97 * baseline_cost`, with inclusive acceptance at precisely 3%. Percent improvement is `100 * (baseline - shift_plan) / baseline`. All six full-episode comparisons are required; no descriptive interval substitutes for a failed check. Whole-row time includes added proposal bookkeeping and evidence recording, so equal candidate counts are not a matched-runtime claim.

The report correctly limits interpretation to one reused engineering case and makes no architecture, general adaptation or statistical-confirmation claim. The reviewer-requested wording clarification is applied in the bound report: the unlaunched item is the prospective fresh-case six-controller qualification, distinct from the already completed six-row engineering exercise.
