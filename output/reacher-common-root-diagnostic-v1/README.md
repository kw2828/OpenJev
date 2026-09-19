# Common-root planning diagnostic

Status at protocol publication: **prepared, not executed**.

This diagnostic separates search effort, horizon and dynamics-dependent ranking on twelve preselected slots from three exposed robot trajectories. The repeated startup states are retained as repeated identities, not independent cases. Every contrast starts from the same copied native state. Short plans hold their final command through step 24, so outcomes describe open-loop branches, not receding-horizon control.

- [Protocol](protocol.json) and [pre-execution arithmetic clarification](report-arithmetic.json).
- [Independent protocol review](protocol-review.md) and [report review](report-review.md).
- [Prepared input receipt](inputs-v1/completed.json): twelve slots, 44 named random streams; no native evaluations.
- [Prior evidence and rationale](../reacher-proposal-memory-engineering-v1/next-decision.md).

The usefulness rule retains the same 3% relative margin, 0.001/action absolute margin, three trajectory checks and eight-of-twelve slot condition. Boundary tests motivated exact comparisons of canonical decimal cost representations; this introduced no tolerance and did not regenerate any input.

The run uses one exclusive attempt, finite execution and replay caps, and separate outcome reporting only after all slots pass independent audit. It is a diagnostic of a conventional supplied-physics planner. It cannot establish a learned recurrent architecture, connectome advantage or scientific generalization.
