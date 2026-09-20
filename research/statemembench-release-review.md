# StateMemBench release and task-interface review

Reviewed 20 September 2026. StateMemBench is a relevant source lead, but this review did not verify an official executable code or dataset release. It is not ready for an OpenJev run on the evidence checked here.

| Artifact | Verified identity and release status |
| --- | --- |
| Paper | [Can Agent Memory Systems Track Evolving State?](https://arxiv.org/abs/2608.19652v1), Xinyi Fan et al.; `2608.19652v1`, submitted 20 August 2026 at 05:41:23 UTC. Only v1 was listed. |
| Paper license | [arXiv non-exclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html). This does not establish a code or dataset reuse license. |
| Code | No author-linked repository verified. No repository revision or code license can therefore be reported. |
| Dataset | No author-linked dataset verified. No dataset revision, split manifest or dataset license can therefore be reported. |

The paper says it releases StateMemBench. However, its checked abstract and main benchmark sections supplied no release URL. The [GitHub repository search](https://api.github.com/search/repositories?q=StateMemBench) returned zero results, as did [Hugging Face dataset search](https://huggingface.co/api/datasets?search=StateMemBench). Broader StateMem and paper-ID metadata searches did not establish an official match. Search coverage is incomplete and time-sensitive; these findings do not prove that no release exists. Search metadata and scope are recorded in the [receipt](../output/statemembench-source-review-v1/source-01/receipt.json).

The paper describes 234 scenarios and 322 graded probes, with multi-session updates and anti-update controls. Its final-question answer pools contain 3–4 options and are hidden from the answering system. They support grading into current-state, superseded-state and other outcomes. StateMem extracts units from each turn and maintains dependencies and supersession across sessions. These are paper descriptions, not implementation checks. [Benchmark and method sections](https://arxiv.org/html/2608.19652v1#S4)

For OpenJev, the original hidden-pool task could test model-maintained state without supplying gold previous state. Exposing that pool to a constrained candidate decoder would create a separate adapted task, not a replication of the published interface. Any later comparison should preserve identical visible history and query information, match memory and inference costs, and include full-context and explicit state-update controls. This is a prospective recommendation, not an admitted experiment.

The concrete next prerequisite is an author-linked release with immutable revisions, licenses, split membership and an inspectable actor/grader boundary. Keep benchmark content unopened until that boundary is established. This review accessed paper descriptions and release metadata only; it did not download benchmark files, inspect case or prompt appendices, run models, copy implementation, or inspect OpenJev's active inference outputs. Public search snippets are not a sealed benchmark boundary, so this is not a claim of globally untouched evaluation material.
