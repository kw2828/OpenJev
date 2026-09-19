# Does recurrent memory outperform an explicit angle cache?

This is a prospective follow-up to the [completed memory comparison](reacher-memory-ablation.md). That study passed its continuation rule, but its small, inconsistent advantage over a packet MLP left a simpler explanation open: retaining the last observed angles may be sufficient. No result from the new comparison is claimed here.

The [protocol](../evidence/reacher-cache-ablation-v1/protocol/plan.json) is frozen at SHA-256 `7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa`. The single scored run has started. Before launch, 351 component tests passed and the complete 15-fit, 60-row engineering rehearsal passed its independent saved-output audit. These checks establish implementation readiness, not effectiveness.

The new study compares five models, with three independently initialized fits per model:

| Model | Information retained across real decisions | Parameters |
| --- | --- | ---: |
| Persistent GRU | Learned recurrent state | 36,805 |
| Current GRU with unconditional encoding | Current sanitized packet only | 36,805 |
| Cached GRU | Last actually observed angles, plus their age | 36,805 |
| Current MLP | Current sanitized packet only | 36,599 |
| Cached MLP | Last actually observed angles, plus their age | 36,599 |

The two reset GRUs run the same observation encoder at every real decision, including when sensing is missing. This separates the information in the cache from the old current-GRU baseline's rule of skipping that update on missing packets. The old gated baseline remains historical context; this experiment does not estimate the isolated effect of removing its gate.

Cache entries come exclusively from real, visible public measurements. Cached values enter internal encoder features without changing the actual packet's missingness, timestamp, or training masks. Predicted angles and learned hidden state cannot update the real cache. Multi-step planning branches remain separate from the next real decision.

All models use the same inherited 768 training episodes, loss, optimizer, minibatch order, 48 epochs, and 1,152 updates per fit. Starting tensors are identical within each GRU or MLP family. Equal parameters and updates do not imply equal compute: cache construction, validation, copying, training, and deployment work are charged and reported.

Evaluation uses 64 fresh paired cases under full sensing, six-step gaps, and ten-step gaps, plus 96 fresh prediction episodes. Full sensing still omits velocity. The models share initial CEM proposals and random innovations; later proposals adapt to each model's scores. Every fit and all 60 controller rows are retained. Supplied-physics, particle, public-kinematic, zero-action and random-action references remain separate competence comparisons.

The frozen primary rule requires persistent recurrence to improve mean native control cost by at least 3% over **both** cached model families on **each** gap panel, with no losing paired fit. Full-sensing mean cost may worsen by at most 2% against either cache. Each persistent fit and the designated physics references must also beat the zero-action floor by at least 10% on both gap panels. These are 28 required checks, not a statistical significance test.

Cache-versus-current contrasts are reported separately for each model family. A failed superiority rule does not prove equivalence. If explicit caching performs as well or better, the result redirects architecture work toward a stronger baseline rather than supporting a recurrent-memory contribution. Any residual advantage still needs a second environment and matched biological-topology controls before supporting a novelty claim.

The code lives in separate `reacher_cache_*` modules. All 58 sources bound by the completed memory study remain unchanged. Engineering tests and a small rehearsal precede a separate protocol freeze and one scored execution. No old checkpoint is reinterpreted as a new model, and no selected fit or episode is substituted after seeing results.
