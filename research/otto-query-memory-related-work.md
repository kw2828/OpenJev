# What the query-memory experiment can establish

The core update has close prior art. The current experiment tests whether sparse
teacher-score residuals improve decisions through a small memory around a frozen
recurrent predictor. It does not establish a new general learning rule.

| Primary source | Relevant prior work | Difference in this experiment |
| --- | --- | --- |
| [Schlag, Irie and Schmidhuber, 2021, section 4.2](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf) | Differentiable fast-weight delta updates; keys and values can encode preceding context. | The target is an externally observed teacher-score residual. Only genuine later queries write; the normalized step size is fixed. |
| [Gated Delta Networks, sections 3.1 and 3.3](https://arxiv.org/html/2412.06464v1) | Delta updates with learned forgetting, normalized keys and queries, and short-convolution history. | The registered model uses an EMA of projected GRU features, tied read/write cues and no forgetting. Memory branches train only a 224-parameter key projection. |
| [Titans, section 3.1](https://arxiv.org/html/2501.00663v1) | Associative-loss updates, adaptive forgetting, and momentum over memory gradients. | Our EMA averages keys, not gradients. The correction matrix receives scheduled external supervision and never feeds the GRU. |

Our equation-level comparison: zero-initialized memory, centered residuals and
centered reads preserve zero column means in exact arithmetic. The update is
therefore a normalized delta rule on the centered score subspace. Its EMA changes
the address used by that rule. Neither this observation nor local contraction
proves better decisions, biological learning or architectural novelty.

If the [frozen comparison](otto-query-memory-protocol.md) fails, one separate
follow-up could test stale-memory interference. Register fresh stable and
causally switching cases, then compare unchanged memory, fixed decay and a small
learned forgetting gate. Match teacher access, training exposure and compute;
retain ordinary recurrent and last-error controls. Require the learned gate to
beat both fixed alternatives after switches without worsening stable cases.
If fixed decay explains the improvement, or the gate suppresses memory everywhere,
reject the adaptive-memory explanation. This proposal does not change the current
experiment or admit another run.
