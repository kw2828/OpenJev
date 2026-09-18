# Reacher world-model qualification

Continuation gate: **FAIL**.

Replayed 124,800 saved native transitions. 9 fixed final fits; no new model, policy or planning calls.

| Panel | Arm | Mean episode cost |
|---|---|---:|
| full | history-211 | 10.072147 |
| full | history-223 | 9.967473 |
| full | history-239 | 9.900662 |
| full | gru-211 | 9.967305 |
| full | gru-223 | 9.901212 |
| full | gru-239 | 9.910809 |
| full | rssm-211 | 9.946795 |
| full | rssm-223 | 11.301243 |
| full | rssm-239 | 12.002651 |
| full | known_state | 6.503609 |
| full | particle | 6.593847 |
| full | zero | 9.899753 |
| full | uniform | 42.081612 |
| ordinary | history-211 | 10.138350 |
| ordinary | history-223 | 9.947876 |
| ordinary | history-239 | 9.989180 |
| ordinary | gru-211 | 10.009345 |
| ordinary | gru-223 | 9.901212 |
| ordinary | gru-239 | 9.910809 |
| ordinary | rssm-211 | 11.515959 |
| ordinary | rssm-223 | 11.280914 |
| ordinary | rssm-239 | 11.702238 |
| ordinary | gru-211-reset | 9.966040 |
| ordinary | gru-223-reset | 9.901212 |
| ordinary | gru-239-reset | 9.910809 |
| ordinary | rssm-211-reset | 12.001981 |
| ordinary | rssm-223-reset | 10.991655 |
| ordinary | rssm-239-reset | 9.644016 |
| ordinary | known_state | 6.503609 |
| ordinary | particle | 6.651602 |
| ordinary | zero | 9.899753 |
| ordinary | uniform | 42.081612 |
| shift | history-211 | 10.140755 |
| shift | history-223 | 10.044042 |
| shift | history-239 | 10.375639 |
| shift | gru-211 | 10.006800 |
| shift | gru-223 | 9.901212 |
| shift | gru-239 | 9.910809 |
| shift | rssm-211 | 11.986717 |
| shift | rssm-223 | 11.326174 |
| shift | rssm-239 | 11.424621 |
| shift | gru-211-reset | 9.964156 |
| shift | gru-223-reset | 9.901212 |
| shift | gru-239-reset | 9.910809 |
| shift | rssm-211-reset | 12.016289 |
| shift | rssm-223-reset | 10.401048 |
| shift | rssm-239-reset | 8.806022 |
| shift | known_state | 6.503609 |
| shift | particle | 6.689564 |
| shift | zero | 9.899753 |
| shift | uniform | 42.081612 |

All episode costs, fit curves, prediction masks, case-paired descriptive intervals and every continuation check are in `summary.json`.

- Source/runtime/plan were authenticated by the caller. Hashes and finite weights authenticate saved training artifacts; training and learned predictions are not independently rerun.
- One/five-step primary angle MSE averages four components only at valid endpoint observations. Five-step additionally reports valid-root/endpoint restriction. Missing-angle truth is audit-only and excluded from model inputs and training angle targets.
- Bootstrap resamples paired episodes after averaging the saved fits. It is conditional on these fits, not architecture-level uncertainty.
- Decision times include batch preparation, filtering and planning; per-case figures are amortized batch costs, not individual response deadlines. Setup is separately measured and charged to whole-run time; learned model construction is included in fit time.
- Zero/uniform candidate scores are zero placeholders, not model predictions. Supplied-physics references have model-class privilege; known_state additionally has current simulator state.
- The fixed final checkpoints and frozen continuation criteria are a pilot qualification, not biological-topology evidence or an ICLR result.
