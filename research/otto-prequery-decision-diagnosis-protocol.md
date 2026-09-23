# Retrospective decomposition of the prequery decision tradeoff

This is a saved-output diagnosis of the completed [prequery study](otto-prequery-calibration-results.md), whose FAIL51/55 decision stays closed. The previously exposed VALID results motivate this analysis. It is not a new held-out test, model selection, or a significance test.

Before decoding prediction arrays, freeze the diagnostic source, fabricated tests, this protocol and an explicit input manifest. Use the original training receipt, summary, validation windows, all twelve saved prediction files, collection identities and successful training/audit parent receipts. Every input must match its original published hash. No checkpoint is loaded, no model or environment runs, and no training label or new teacher query is generated.

Compare auxiliary minus original-MSE within each of the two architectures, retaining all three paired seeds. Both sensing settings, all six originating VALID cases per setting and all three collector paths per case remain. The six comparisons are fixed in advance. Do not pick a fit, near-tie threshold, episode-length cutoff or favorable setting after inspecting the arrays.

For each episode, partition its eligible nonquery rows into initial steps1-3 and postcorrection steps5 onward. Query steps and padding never score. With counts nI, nP and n=nI+nP, the full metric difference is:

`delta_full = mean_over_all_episodes[(nI/n)*delta_initial_mean + (nP/n)*delta_post_mean]`.

Empty phases contribute zero; every declared episode remains in the outer denominator. The published primary difference instead averages each episode's postcorrection mean. Consequently:

`delta_full = delta_primary + initial_contribution + post_reweight_contribution`.

Report both additive contributions and the separately weighted full and primary metrics. Never interpret full minus primary as an initial-period effect. Agreement and teacher-score gap use exactly the existing first-legal strict float32 near-minimum rule at 1e-10, with gap subtraction in float64 from original float32 scores.

Within each phase, retain all four paired correctness transitions: correct-to-correct, correct-to-wrong, wrong-to-correct and wrong-to-wrong. Their masses and gaps use each episode's full nonquery denominator, so they sum to the original full metric difference. Retain raw counts, both gaps and signed changes; near-minimum agreement does not require a numerically zero gap. Preserve every zero-valued transition cell.

Outputs include every episode pair, phase and transition cell; all per-seed overall, setting, case and collector aggregates; and equally weighted means over the three fit seeds. The analysis must reconcile original full and primary agreement/gap metrics and the additive identities, with explicit numerical residuals. These are correctness checks, not a new scientific continuation rule. A separate reader will check scalar arithmetic against the saved episode and transition tables.

Execution is one local process with a 120-second original supervisor, one numerical thread, a 1 GiB worker RSS limit and 16 MiB output limit. It uses a new exclusive output directory, retains failure records if interrupted, and never mutates the earlier study. Qualification uses fabricated inputs only. Publication may render these saved aggregates but may not generate scientific predictions.

The immediate decision is whether the observed full-trajectory loss is explained by initial-period changes, episode reweighting, or both, and how correctness transitions relate to score-gap gains. The result can motivate a fresh experiment; it cannot reverse the original failure or establish gradient conflict. A separate prior readout would add 120 or 116 parameters to the current models and also alter the correction signal. Shared versus separate readouts and main-task gradient protection are established approaches, so neither is itself a novelty claim.
