# Portable features and matched ensembles

This prospective study was frozen at `6402a29` before any of its new simulation runs. It follows the preserved failures and partial gains in the [ammo-only](cadence-study.md) and [combined-event](event-cadence-study.md) studies.

The five historical history-MAP models were trained on disjoint Center data. Their Line outcomes vary substantially, and their learned ammo coefficients differ. This suggests testing a smaller targeting representation without absolute ammo, health or distance levels. It is a hypothesis about this dataset, not proof that these variables are universally irrelevant.

## Matched training and controls

Every ensemble uses the same total 80 historical training episodes, divided into the same five fitting replicates with identical issued-fire labels. Original-history ensembles reuse the preserved 17-feature MAP fits. The new current head has four inputs: bias, target visibility, capped aim error divided by a target-width tolerance, and target width. Its history version adds previous fire, previous hit, time since last issued fire, and a history-present indicator, for eight inputs. The Gaussian prior and optimizer are unchanged.

Each ensemble averages five probabilities, not five sets of generated text. This is ordinary model averaging, not a new Bayesian procedure. Both feature removal and representation geometry change, so the comparison cannot isolate one input as the cause of any gain.

The eight development arms are rules, rules with event memory, original-history ensembles with and without event memory, and both new heads with and without event memory. All ensemble arms require a visible target; the untouched single-fit historical references retain their original behavior. All share the same rule steering, seven-tic windows, horizon and command cost. Event memory is the fixed one-window rest after an ammo decrease or preceding hit.

Development uses 24 new seeds in each scenario, 384 episodes. Eligible candidates must gain at least 0.5 mean net utility while losing no more than 0.5 mean kills against rules in each scenario. Select the candidate with the largest minimum gain across scenarios, once. No fallback is allowed after confirmation.

One confirmation, if authorized by development, uses 96 fresh seeds per scenario and ten arms counting the five historical fits separately: selected candidate, rules, original ensemble, event-memory rule, selected head with its event gate disabled, and five original history-MAP policies. It uses the same primary utility and kill criteria as the earlier cadence protocols, including adjusted bootstrap intervals and the latency screen.

The original ensemble is an essential control: gains against individual models alone could come from averaging and access to all five training replicates. Secondary comparisons against the original ensemble and against the same head without the event gate determine what the component comparisons support. These secondary results cannot rescue a failed primary gate.

Both scenarios appear in development. Confirmation measures new-seed performance, not unseen-task generalization. Intervals are adjusted within each study, not across the entire sequence of adaptive research studies. All attempts are retained, and no ICLR novelty or second-environment claim follows from these comparisons.

## Reproduce

```sh
uv run python research/portable_head_doom.py --stage development --output evidence/portable-head-doom-v1/development-new
uv run python research/portable_head_doom.py --stage confirmation --development evidence/portable-head-doom-v1/development-new --output evidence/portable-head-doom-v1/confirmation-new
uv run python research/analyze_portable_head_doom.py evidence/portable-head-doom-v1/confirmation-new --output evidence/portable-head-doom-v1/analysis-new.json
```

Use new output paths. Source, protocol, fitted models and run outputs are hashed. No paid API is used.
