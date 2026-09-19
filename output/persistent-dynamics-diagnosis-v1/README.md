# Why did the accurate-state reference perform worse?

**No concrete implementation bug was found.** The saved evidence supports a search-adequacy experiment before a neural adaptation study. It does not establish which approximation caused the poor result.

- The public observer exactly matches the intended causal equations, with no angle aliasing. Its current velocity rule has lower whole-history error than three fixed alternatives on all five planning histories. [Observer diagnosis](observer-results.md).
- Gain estimation drifts to the upper endpoint before the gain actually changes. Its closer post-change estimate is therefore insufficient evidence of successful adaptation. No ranking or window-indexing defect was found.
- One-step distance prediction RMSE is **0.00024-0.00043 meters** across the five planning roles. The true-state reference has the lowest prediction error yet worse native control cost. Correct one-step predictions do not guarantee useful planning. [Candidate calculations](candidate-results-01/results.json).
- No candidate reward was clipped. Native cached-body distance and final-angle geometry differ, but their one-step RMSE is **0.000015-0.000110 meters** on the selected trajectories. These discrepancies alone do not explain the control reversal.
- During the first 50 actions, known-gain and true-state planning chose the zero-action anchor **42% and 32%** of the time, versus **12%** for nominal planning. The known-gain model predicted much less progress within its short horizon. This makes search and horizon limitations plausible, not proven.

The [contract review](planner-contract-review.md) also retains the principal modeling limitation: candidate motion is deterministic and noiseless, while only the action penalty is averaged over noise. The planner does not optimize expected noisy distance. Noisy dynamics, finite search, a short horizon and closed-loop path dependence remain possible explanations.

All calculations use completed saved traces from the same engineering seed410 case. No new simulation, random draw, model inference or training was performed for the diagnosis. Existing sources and failed study criteria remain unchanged.

## Next bounded comparison

[Compare cold search, last-action centering and shifted-plan centering](planner-followup-options.md), using the same model, score, horizon, innovations and 256-candidate budget. The repeat-last-action control tests whether retaining an entire plan adds anything beyond ordinary action persistence. This is an established MPC mechanism, not architectural novelty.

The separately recorded [nine-row protocol](../reacher-proposal-memory-engineering-v1/protocol.json) crosses these three modes with nominal, known-current-gain and known-current-state/gain roles. Cold rows must reproduce the earlier trajectories exactly. All rows and replay must complete before interpreting performance. No fresh scientific seeds or neural fits are allocated.
