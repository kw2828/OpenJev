# Does persistent memory help once scoring is held constant?

**Status: frozen, execution started, no scientific result yet.** The protocol was pushed before evaluation. Results require the complete independent audit; partial rows will not be used to select methods or change thresholds.

The [completed geometry intervention](reacher-geometry-score.md) improved control with unchanged model weights. This next experiment tests the already-trained memory controls under that same scoring rule.

| Model | Information carried across real decisions |
|---|---|
| Persistent GRU | Learned state updated by observations and issued actions |
| Current-packet GRU | Hidden state resets at each observation packet |
| Cached-observation GRU | Hidden state resets; input retains the last observed angles and their age |
| Cached-observation MLP | The same observation cache, with a feedforward predictor |

All three GRU families share the same parameter count and paired initial weights, but have separately trained final weights. They all remain recurrent within imagined rollouts. This tests the trained persistent update policy against the declared reset controls, including their different observation-assimilation rules.

We retain all three fits of every family and evaluate 64 fresh paired cases with full sensing, six-step gaps and ten-step gaps. No new training or model selection occurs. Every learned controller and all three physics references use geometry scoring and CEM256 search. Known-state physics alone receives the true current state; other controllers retain their declared public-input boundaries. Equal search budgets do not imply equal total computation.

The frozen 25-check rule requires persistent GRU to improve mean cost by at least 3% against both reset GRUs on both gap panels, lose no paired fit on those comparisons, and satisfy the full-sensing and zero-action competence checks. The cached MLP comparison is secondary. Failure of any required check prevents a passing conclusion.

Execution has a fixed 5,700-second limit; independent audit has a separate 3,300-second limit. The audit must check all 163,200 executed transitions and replay 78,741,504 nominal physics candidate transitions plus 28,800 selected advances. Failed or incomplete attempts remain evidence.

Even a pass would not establish a novel architecture or biological advantage. A stronger explicit history baseline remains necessary. Conditional case intervals also do not quantify uncertainty over new training datasets or training-seed populations.

- [Frozen protocol](../evidence/reacher-geometry-memory-v1/protocol/plan.json)
- [Readiness receipt](../evidence/reacher-geometry-memory-v1/protocol/readiness.json)
- [Independent freeze review](../output/reacher-geometry-memory-v1/freeze-review.md)
- [Capacity measurements and limits](../output/reacher-geometry-memory-v1/readiness-review.md)
- [Launch receipt](../evidence/reacher-geometry-memory-v1/launch.json)
