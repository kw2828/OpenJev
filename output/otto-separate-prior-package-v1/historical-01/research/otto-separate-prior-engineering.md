# Separate forecast and decision readouts

The [saved-output diagnosis](otto-prequery-decision-diagnosis-results.md) identifies a specific tradeoff: the auxiliary forecast objective improves later decisions on average, but worsens the first three steps. The new module gives prior forecasts their own output layer. **It has not been trained or evaluated for effectiveness.**

## Mechanism

The [implementation](../src/openjev/research/otto_separate_prior_scores.py) extends the pinned prequery adapter. The recurrent state, public features, genuine planner queries, correction update, padding and detached-carry interface are inherited. A new `prior_output` layer supplies the four score forecasts made before later queries. The existing `output` layer still scores nonquery actions.

| Recurrent family | Shared readout | Separate readouts | Added parameters |
| --- | ---: | ---: | ---: |
| Explicit error correction | 5,978 | 6,098 | 120 |
| Ordinary error-fed GRU | 5,996 | 6,112 | 116 |

Both output layers start at zero. Existing parameters retain exactly the shared model's initialization for a paired seed. Copying the decision layer into the prior layer recovers the shared model's forward values and recurrent call sequence. The extra parameters do not introduce an extra readout invocation: they replace the layer used at an existing later-query call.

This is a small intervention, with two important effects. It separates the direct readout parameters used by the two objectives, and it changes the prediction used to compute the correction signal. The recurrent backbone remains shared. Later decision losses can still reach the prior layer through correction. Consequently, separate readouts do not guarantee protection of early decisions after training, and a gain would not by itself establish gradient conflict as the cause.

The adapter relies on a pinned internal contract: rank-two readouts occur only at later queries, while rank-three readouts score nonquery sequences. It rejects calls outside an active inherited forward. A future base implementation change requires requalification of that contract.

## Qualification

The tests use fabricated score histories only. Qualification covers seeded initialization and parameter counts, copied-head forward equivalence, an independently expanded GRU step, current/future-query causality, direct and correction-mediated gradients, padding and ended lanes, chunked carry, detached gradients, malformed input, input/carry ownership and frozen output-container fields. In the copied-head gradient check, the sum of the two separate readout gradients matches the shared readout gradient. Source pins bind the new module, tests and inherited dependencies.

**All 16 fabricated cases pass**, and Ruff passes, in the [first runtime qualification](../output/otto-separate-prior-engineering-v1/qualification-01/receipt.json). The [original terminal record](../output/otto-separate-prior-engineering-v1/qualification-01.terminal.json) retains process closure, and the [independent source review](../output/otto-separate-prior-engineering-v1/source-review-01.json) records the internal-contract checks and exact pins. A preparation request was superseded before execution to strengthen the tests; both requests remain in the engineering directory. This evidence concerns implementation behavior only. There are no new saved fits, empirical predictions, teacher queries or simulator episodes in this engineering step.

## Prospective comparison

The next study should test the complete two-by-two combination of shared/separate readouts and original/auxiliary losses, in both recurrent families, with three paired fit seeds: **24 fits**. Including separate-readout original-MSE controls lets us distinguish the effect of changing the readout/correction path from the added forecast objective. Separate readouts still add parameters, so this is not an exact parameter-matched architecture comparison.

Hold training trajectories, sampled windows, optimizer updates, query schedule and fit seeds matched. Measure total training and inference cost, including all forecast and correction work; equal update counts do not imply equal computation. All four readout/loss cells in each family and all three seeds must be reported.

The main comparison is separate-auxiliary versus shared-auxiliary. Assess whether it recovers initial and full-trajectory agreement while retaining the forecast and later teacher-gap improvements. Also compare against original-MSE controls and test whether the readout-by-loss interaction differs between explicit correction and ordinary GRU. A result that benefits both families should be reported as an objective/readout improvement, not an explicit-correction advantage.

Before running, freeze a new empirical protocol with unused collection and fit seeds, exact budgets, endpoints, effect margins and continuation criteria. The exposed cases that motivated this design are development data. Fresh validation and a predeclared scenario shift are necessary before a later autonomous utility-versus-compute comparison. This design note is not the frozen empirical protocol and admits no training run on its own.

Readout separation has substantial prior art, including [Cross-stitch Networks](https://arxiv.org/abs/1604.03539). [Gradient similarity](https://arxiv.org/abs/1812.02224) and [PCGrad](https://arxiv.org/abs/2001.06782) offer distinct future controls if the tradeoff persists. None of these ingredients alone establishes novelty, biological wiring, or an ICLR-ready result.
