# Query-gate learning evidence

Completed: 60 collection trajectories, six final fitted gates, 576 autonomous
evaluation episodes and an independent saved-record audit. The continuation
rule failed: 30/38 required conditions passed. All 18,012 learned gate decisions
queried the fixed neural planner. No computation saving or recurrence advantage
was established.

[Full results](../../research/otto-query-gate-learning-results.md) describe the
protocol, costs, failures and audit limits.

The repository keeps compact plans, summaries, receipts, figures and initial/final
gate checkpoints. Larger journals, trajectory records, TRAIN arrays and parity
witnesses are in the [complete phase archive](https://github.com/kw2828/OpenJev/releases/tag/otto-query-gate-learning-v1).
The archive restores repository-relative paths when extracted at the repository
root. Its manifest lists every included file's SHA256 and size; `SHA256SUMS`
identifies the archive and manifest.

This archive covers this study's phases and execution sources. Original OTTO
weights, native runtime, earlier qualification records and other inherited
dependencies remain identified by the frozen plans and previous releases. It is
not a standalone runtime bundle. Strict runtime authenticators also retain the
original absolute paths; relocating files alone does not qualify a rerun.

The original failed pin-transcription check remains recorded inside
`learning-publication-check-01.json`. No scientific phase was retried. Checkpoints
are research artifacts, not promoted controllers.
