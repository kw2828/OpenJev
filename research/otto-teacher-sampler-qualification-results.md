# Teacher sampler matches the native generator on the fixed cases

**All 461 fixed comparison cases passed**, including 408 native simulator steps. The sampler and the original seeded simulator agreed exactly on the tested probability vectors, cumulative probabilities, selected categories, public observations and posterior updates. No tolerance, probability repair or replacement case was used. This qualifies the tested generation mechanics; it does not establish better labels, learning or an architecture advantage.

[Protocol](otto-teacher-sampler-qualification-protocol.md) · [Frozen plan](../output/otto-teacher-sampler-qualification-v1/plan-01.json) · [Summary](../output/otto-teacher-sampler-qualification-v1/run-01/summary.json) · [Original execution witness](../output/otto-teacher-sampler-qualification-v1/execution-witness-01.json)

## What ran

The fixed schedule used sensing lengths 3, 4 and 5 on the 53x53 grid. Twelve nonfound geometries per setting each received eleven uniforms, including values immediately below, at and above categorical boundaries. Four additional moves per setting found the source immediately. The comparison included blocked boundary moves as transport fixtures; the continuation sampler still permits only in-bounds first actions.

| Work | Completed |
| --- | ---: |
| Native constructions, including normal reset | 3 |
| Additional resets | 0 |
| Native steps | 408 |
| Nonfound / immediate-found steps | 396 / 12 |
| Additional categorical probes | 44 |
| Additional source probes | 9 |
| Actual native hit / source draws | 440 / 12 |
| Public snapshot initializations / updates | 408 / 408 |
| Teacher choices, model calls, fits or training-label collection | 0 |

The 12 source draws include the three initialization draws, which were retained as evidence and never used as fixture sources. Hidden sources, random tapes and draw evidence stayed with the evaluator. Snapshots received only public packets, public beliefs and the known observation kernel. The final native and public posteriors were saved for every step. All twelve terminal states matched an independently constructed point mass, with no odor draw on discovery.

The original worker completed in **9.028 seconds**, and the supervising process completed in **9.169 seconds**, including process cleanup. Peak worker RSS was **105,250,816 bytes**. The completed run directory contained **23,152,533 bytes**. These are qualification costs, not policy or training-label throughput.

## Review and failure handling

The paired sampler had already passed [54 fabricated tests](otto-teacher-rollouts-component.md). Independent source review then found a logging gap: a random draw could finish before its enclosing native step or constructor failed. Before freezing this run, a transparent evaluator wrapper was added around the original bound draw method. It delegates the unchanged sampler exactly once and saves nested attempts and returns. The protocol was amended before native execution; its numerical cases and acceptance rule did not change.

Five new fabricated regression tests passed for constructor failure after a source draw, step failure after a hit draw, a failed nested draw, normal closure and a resource failure after a completed draw. The first engineering check failed Ruff on two intentionally broad failure-preservation handlers. Only explanatory lint annotations changed. The executable AST remained identical, and the final lint check passed; the five tests were not needlessly repeated.

- [Original engineering receipt](../output/otto-teacher-sampler-qualification-v1/engineering-01/receipt.json): tool `37a056`, exit 1; five tests passed, two lint findings retained.
- [Annotation-only correction receipt](../output/otto-teacher-sampler-qualification-v1/engineering-02/receipt.json): tool `a31314`, exit 0; unchanged executable AST and passing Ruff.
- [Pre-execution admission record](../output/otto-teacher-sampler-qualification-v1/execution-admission-01.json): fixed one-run allocation, with no native calls before freezing.
- [Worker receipt](../output/otto-teacher-sampler-qualification-v1/run-01/receipt.json): `859bfb303daea27d187670b0e3e452855f2fd3be7ce5824c371a0526879268b6`.
- [Original parent terminal](../output/otto-teacher-sampler-qualification-v1/supervision-01.terminal.json): `785df84e5a9839903cd50f9b8b3e7d8a1a14ed21ec1c48e6230e41c723cb0c51`; original session `63421`, terminal tool `f9d7e4`, exit 0.

Root verification authenticated every saved payload, source and input, matched all attempted/returned counts and confirmed clean process termination. All 219 frozen spatial-study source hashes remained unchanged.

The [independent saved-evidence review](../output/otto-teacher-sampler-qualification-v1/independent-review-01.json) passed **38,196 checks with zero failures**. It authenticated 17 sources, 13 inputs and all 417 worker payloads; reconciled 7,420 journal entries and 3,710 completed operations; independently recomputed categorical boundaries and source selection; and reconstructed every posterior update from the saved public inputs and kernels. All 408 posterior pairs agreed, including twelve terminal point masses. It used no native simulator, sampler or model imports or calls. Original review tool `cfc345`, exit 0; receipt SHA-256 `e1a1974f4e58b7844bc11d4e90549b9b14a09cadb62df9e1c6ef9b7febdd35cd`.

## Next experiment

The prospective [six-anchor pilot](otto-teacher-label-pilot-protocol.md) fixes sixteen paired replicates per TRAIN anchor and the full 2,188-move cap. It will measure capped costs, action differences, censoring, descriptive paired uncertainty and full generation cost. Exact anchor identities, extraction and resource accounting must be qualified and frozen before collection. Passing this mechanical qualification alone does not establish that teacher-continuation supervision improves the same ordinary policy; the [training comparison](otto-action-cost-followup.md) remains separate.
