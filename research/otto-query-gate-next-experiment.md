# Next experiment: learn the value of a planner query

**Saved-data diagnosis and a prospective proposal only.** No new inference, fitting, environment calls or threshold search was performed for this note. The completed study remains a failed continuation result: **30/38 required conditions passed**. A new protocol, runtime qualification, data split, seed reservation and budget must precede any follow-up.

## What the saved TRAIN predictions establish

All six fitted gates requested the planner on **every one of their 1,920 saved TRAIN decisions**, before considering autonomous evaluation. The fixed query threshold was 0.05. Every saved probability exceeded it:

| Gate | Minimum TRAIN probability | Median TRAIN probability | Requests / TRAIN decisions |
|---|---:|---:|---:|
| GRU32, seed 40101 | 0.165262 | 0.165409 | 1,920 / 1,920 |
| MLP190, seed 40101 | 0.097103 | 0.148753 | 1,920 / 1,920 |
| GRU32, seed 40102 | 0.163728 | 0.163891 | 1,920 / 1,920 |
| MLP190, seed 40102 | 0.102775 | 0.154696 | 1,920 / 1,920 |
| GRU32, seed 40103 | 0.164062 | 0.164250 | 1,920 / 1,920 |
| MLP190, seed 40103 | 0.094165 | 0.147950 | 1,920 / 1,920 |

The dataset contains **326 positive labels out of 1,920 rows**, or 16.9792%. Equal-episode weighting gives **17.7504%**; the six fitted weighted mean probabilities are 17.57% to 18.32%. For this reporting diagnostic, rounded float32 episode weights are normalized by their sum. The original training objective and its denominator are unchanged.

The model moved away from its initial probability of 0.95, but the resulting probabilities all remained above 0.05. The TRAIN witnesses establish that the all-query outcome was already present on the fitted data. They do not establish probability calibration, optimal thresholds, or that recurrence is intrinsically ineffective. The original Torch/NumPy qualification passed; this diagnosis reads the saved NumPy witnesses rather than rerunning either backend.

The target is **disagreement with the fixed neural scorer**: whether the analytic action lies outside its eligible float32 near-minimum set. It does not measure discovery improvement, expected time saved, or whether a query is worth its cost. Even the saved continuous gap is in neural score units, not observed extra moves.

In the closed evaluation, all **18,012 learned decisions** queried. The learned gates reproduced the neural control's moves and added overhead. All 576 episodes succeeded, but all six primary GRU cost checks and both recurrent-versus-MLP cost checks failed. On the declared length-5 transfer probe, the neural control used **169.30 weighted moves versus 72.47 for analytic**. Thus copying the teacher can be actively unhelpful under this shift despite eventual success. These cases are now development evidence and cannot become untouched confirmation cases for a revision.

## A precise candidate: a budgeted value-of-query gate

Use the unchanged analytic and neural action endpoints. Replace the disagreement target with a **paired, short-horizon intervention advantage**, then test whether recurrence predicts that advantage better than simpler histories.

1. On a newly frozen TRAIN cohort, select public anchor states by a deterministic rule that does not inspect scores, labels, or outcomes. Retain their complete preceding public histories. Allocate labels equally across episodes; do not select only disagreements or successful paths.
2. At each anchor, sample **16 sources from the public posterior**. For each source, compare two branches: force the analytic action or the neural action once, then use the same analytic continuation policy for the remaining **32-move horizon**. Couple the source and indexed observation uniforms across branches while preserving each branch's conditional observation law. Count the first forced move, stop at discovery, and retain capped outcomes.
3. Define the label as the paired mean of `capped_moves_skip - capped_moves_query`. Positive values favor querying. Save individual paired returns and their uncertainty. This is a local advantage under one declared continuation policy, not an optimal value function or a full-episode regret guarantee. A 32-step cap can hide benefits or harms beyond that horizon; report the capped fraction.
4. Fit matched GRU and MLP regressors to that same target, with identical episode weights, optimizer budget, row exposure and final-checkpoint rule. The actor receives only public features and its permitted history. Sources, branch outcomes, current neural scores and target uncertainty remain evaluator-side training information.
5. Add a common online budget: after `t` decisions, total requested queries must satisfy `Q_t <= ceil(t/2)`. Within available credit, query only when the predicted advantage exceeds a threshold fixed using a separate, fresh VALID cohort. Predeclare the threshold candidates and tie rule before VALID; select once and freeze before EVAL. Do not select a threshold from the completed study. The quota controls planner calls, not wall time, so gate and filtering overhead still have to earn their cost.

Before expensive autonomous evaluation, report whether paired advantages have useful variation relative to their sampling uncertainty. Preserve negative and indistinguishable labels. If these local targets contain too little signal, stop this mechanism rather than adding architectural complexity. Any numerical signal criterion must be frozen before reading the new returns.

## Controls and decision rule

The crucial control is the **original disagreement objective with the same prospective VALID procedure and query quota**. It separates a better decision rule from a better target. Add fixed period-two and seeded random-half schedules under the identical quota, plus standalone analytic and neural controls. Include a gate based on public posterior entropy with its threshold fixed by the same prospective VALID rule and its computation charged. Include a finite-history MLP alongside the parameter-matched current-feature MLP and GRU; supply it a fixed eight-step public history, with its added input/compute cost reported. Use three paired fit seeds and complete episode splits throughout.

A recurrent advantage requires beating both MLP controls at comparable total cost, not just beating an MLP that lacks readily available history. The public posterior is already a sufficient state for the specified simulator. Recurrence here can compensate for compression into 31 features or learn useful temporal summaries; it would not, by itself, demonstrate a recurrent world model. Do not call this RL or a biological architecture result without additional implemented mechanisms and matched evidence.

Keep the existing success, move and complete-cost requirements as the starting acceptance standard. Report every fit, paired case block, failed trajectory and capped tail. Define any revised family comparisons prospectively, with one primary configuration and all controls visible. Include query counts, actual controller time, cold setup, shared annotation costs and training amortization. A query quota alone cannot pass the cost criterion.

Collect new primary and transfer cases. Keep length 5 explicitly as a previously observed shift, and add one separately declared transfer condition if claiming broader robustness. Freeze transfer parameters before new labels or evaluation. Teacher disagreement and short-horizon labels can both fail after a dynamics shift; only new autonomous results can establish useful savings or recurrence. No ICLR-level novelty or effectiveness claim follows from the present diagnosis.

## Evidence boundary

The saved-only arithmetic was executed once in tool result `617157`, exit 0. It authenticated the closed evaluation plan and collection/training receipts, then checked the referenced dataset and six parity witnesses before decoding TRAIN arrays. It did not load model checkpoints, inspect EVAL per-decision probabilities, sweep thresholds, or rerun scientific components. The outcome discussion uses the separately closed saved evaluation and independent audit.

- [Collection receipt](../output/otto-query-gate-learning-v1/collection-01/receipt.json): `51709d4791f82b157d8b519aee5f129fa5df2b4af091cb854b282c0acbf256cf`
- [Training receipt and parity witnesses](../output/otto-query-gate-learning-v1/training-01/receipt.json): `ea97a73b687c8fb7bbde4ae0e63bed7a7cb8cef26cace154eed5207b0989dd26`
- [Evaluation receipt](../output/otto-query-gate-learning-v1/evaluation-01/receipt.json): `337206eda48e0165db915552affb816e1d328b3c43534b819ea192c9ab512b5b`
- [Saved audit receipt](../output/otto-query-gate-learning-v1/audit-01/receipt.json): `705108876f9a399f53687ff7e8ec8df39d391573e8c7f814e4a4ebe28b8f9be0`
- The [original design](otto-query-gate-learning-design.md) and [frozen evaluation protocol](otto-query-gate-evaluation-protocol.md) retain the original target, threshold and criteria. This note changes neither.
