# Decisions after matched teacher-cost learning

22 September 2026. Outcome-blind draft using published results and the [current protocol](otto-teacher-learning-protocol.md). No live outcomes, empirical arrays, model calls or simulations were used. This changes no criterion and admits no experiment. Each follow-up ends in fresh autonomous evaluation.

Read all **33 conditions**: three analytic positive-control checks, eighteen individual continuation-fit competence checks and twelve family comparisons. Family means cannot replace individual competence. An incomplete run, failed numerical review or failed analytic positive control needs its original failure reported before a performance interpretation. Competence with only a relative-cost failure is also distinct from incompetence.

Prior results discourage another blind backbone change. [Capacity](otto-capacity-results.md) passed 6/6 TRAIN-excess criteria but 0/6 VALID criteria. [Bellman continuation](otto-bellman-control-results.md) passed 19/42 overall and 0/18 competence. [Spatial control](otto-spatial-control-results.md) passed 0/18 candidate and 0/72 other learned competence checks while analytic found all 72 sources. [Student-state pooling](otto-symmetry-head-results.md) already used analytic preferences. The later [coverage quota failure](otto-coverage-collection-results.md) never tested coverage efficacy.

| Closed current result | Next factor | Strong ordinary control |
| --- | --- | --- |
| All 33 pass | Shared iterative spatial computation with competent supervision | Retained competent dense heads and an untied feedforward spatial network |
| Relative action quality improves, but competence fails | One refresh of current-student TRAIN support | Fresh old-only continuation training with identical update allocation |
| Continuation regresses or gives no coherent benefit | Finite-sample targets: 16 versus 64 replicates | Original 16-replicate targets, same anchors/model/scaling |

## Pass: confirmation and one architecture comparison

Reuse the complete frozen continuation-label dataset and retain all three competent dense checkpoints. Test **two iterations of shared local message passing within each decision**. The existing belief grid and deterministic public geometry initialize a four-channel cell state; the same learned self/nearest-neighbor update runs twice; pooling and a readout produce four costs, retaining position, mask, sensing and twenty forecast features. Reset the internal state at every decision.

The key control is the identical two-step spatial network with untied weights, alongside the competent dense reference. Keep targets, global scale, episode weights, eight training views and optimization allocation fixed. Tied and untied networks have equal execution depth and receptive field but different parameter counts; report that distinction.

Six new fits require **2,400 updates**. Nine learned heads plus analytic on 72 fresh full-horizon cases yield **720 episodes**, at most **1,575,360 native steps**. Proposed ceiling: **7,200 seconds, one thread, 4 GiB RSS, 8 GiB output**. Require dense competence to confirm, every candidate seed to meet existing absolute criteria, and prospectively specified move/cost improvement against both controls. No advantage over the untied network falsifies this practical weight-sharing hypothesis.

This is not new scientific territory. [Value Iteration Networks](https://arxiv.org/abs/1602.02867) already embed differentiable planning computation in policies. This proposed cell update is not an exact Bellman operator. The repository's previous spatial trial instead used scalar-return supervision and observation-branch deployment.

## Relative gain without competence: one coverage round

Collect **24 fresh TRAIN cases across the two training kernels**, balanced over initial hits, with all three current continuation heads: **72 trajectories**, at most **157,536 moves**. Retain censored paths. Select up to four deterministic time-spaced pre-action prefixes per trajectory, including reset. Keep every available selected row, up to **288 new anchors**; use no exact row quota or error-based selection. Sixteen teacher replicates over eligible actions require at most **18,432 continuations / 40,329,216 label moves**.

Compare old-plus-new supervision against a fresh old-only control. Use the same ordinary model, three paired initializations, original target scale, episode-balanced sampling rule and exactly **400 batches of 128 per fit**. Both arms use this common sampling recipe; the fresh old-only arm controls its difference from the current epoch-permutation recipe. Reusing the original scale separates coverage from loss magnitude. Retain every labeled row and fit.

Bound six fits at **2,400 updates**, then **504 fresh paired evaluation episodes** under current absolute and relative rules. Proposed ceiling: **10,800 seconds, one thread, 4 GiB RSS, 16 GiB output**. Continued incompetence or no advantage over old-only training rejects this one-round intervention. No automatic second round.

The hypothesis concerns coverage of current decisions by older learners' anchors; failure alone does not establish distribution shift. Interactive cost-sensitive learning is [Ross and Bagnell prior art](https://arxiv.org/abs/1406.5979). Current-student continuation labels are the repository-specific untested factor.

## Regression: one target-precision comparison

Keep the same 558 anchors, teacher, horizon and actions. Reuse replicate IDs **0-15** unchanged and append **16-63** under the same stream namespace. This adds **96,912 continuations**, at most **212,043,456 moves**. Compare fresh paired models on original sixteen-sample means versus combined sixty-four-sample means. Preserve the original sixteen-sample global scale, episode weights and optimizer recipe in both arms.

Retain all ties, censoring, paired standard errors and split-sample disagreements; none filters anchors or becomes a confidence claim. Six fits use **2,400 updates**, followed by **504 fresh evaluation episodes**. Proposed ceiling: **10,800 seconds, one thread, 4 GiB RSS, 32 GiB output**. This is the expensive fallback. Project its cost from the closed current run before admission; if it cannot fit, report that constraint rather than selecting easier states. Better label stability without better control rejects sampling noise as a sufficient practical explanation. Sixty-four samples still guarantee no correct ranking.

## Interpretation

Targets estimate one forced action followed by the analytic teacher, not optimal values or the learned policy's future costs. Precision and coverage are separate hypotheses. Features already use **53 times square-root belief**, and targets already use global RMS scaling. Repeating the old raw-probability gain is not an equivalent missing control; per-panel cost normalization would change supervision across states.

Exact public beliefs and known kernels are supplied. Internal iteration cannot claim temporal-memory recovery. A later recurrent test must separately restrict that input and retain full-belief and ordinary recurrent controls. [V-JEPA 2](https://arxiv.org/abs/2506.09985) establishes action-conditioned latent prediction as prior art, not evidence for this repository's biological-learning or robotics claims.

These ceilings are proposed allocations, not measured runtime or predicted success. Compression may not fit worst-case journals. Charge generation, fitting, full controller work, simulation and saved replay separately. Competent control, mechanism-specific confirmation and another environment remain the evidence needed for the broader goal.
