# OpenJev improvement studies

Three prospective follow-ups ran **6,408 new Doom episodes** after the failed 1,980-episode memory ablation. Each protocol was committed before its development run, selected at most one candidate, and used new confirmation seeds. All attempts and failed gates remain available.

We found a repeatable engineering gain: remembering observed firing events avoids some redundant fire commands. We did **not** establish a method that clears the full research continuation gate or an ICLR contribution.

| Frozen study | New episodes | Confirmed gain over rules | Full gate |
|---|---:|---|---|
| [Ammo-only cadence](cadence-study.md) | 1,944 | Center +0.854 net utility; Line gameplay unchanged | Failed |
| [Combined ammo/hit events](event-cadence-study.md) | 2,160 | Center +0.823; Line +5.435; identical paired kills | Failed |
| [Portable heads and matched ensembles](portable-head-study.md) | 2,304 | Center +2.322; Line +5.663; Line kill noninferiority unresolved | Failed |

Each confirmation uses 96 paired seeds per scenario. Utility counts successful issued-fire windows minus 0.25 per issued-fire window; net utility additionally charges one unit per second of decision compute. The combined-event rule preserved kills in all 192 paired episodes. It reduced command counts, not demonstrated ammunition use or combat performance. This gain depends on the stated command cost.

![Combined event-memory confirmation](../evidence/event-cadence-doom-v1/confirmation-contrasts.png)

The full gate additionally requires utility improvement and kill noninferiority against the five historical history-MAP fits in both scenarios, plus a latency screen. All three studies failed at least one of these requirements. The history fits vary widely on Line. Bootstrap intervals resample seeds and historical fits; their 99.375% levels adjust the eight primary contrasts within each study, not the entire adaptively chosen research sequence.

The final study offers a useful component check: the **same learned head** gains +0.760 utility in Center and +4.846 in Line after adding event memory, with kills identical in all 192 paired episodes. These are exploratory secondary comparisons. The portable head beats the matched original ensemble on Center utility but not on Line. We cannot credit Bayes, learned recurrent state, or a new architecture for the observed improvements.

## Decision

Publish the verified command-efficiency result and retain the failed gates. Do not advance to an adaptive-compute efficacy claim or a second-environment superiority claim on this evidence. A next study needs a stronger target than reducing redundant commands: for example, learning hidden weapon readiness with randomized-action labels, then testing whether it improves kills or survival at a fixed observation/action budget. It would need fresh data, untouched seeds, simple cooldown controls, and an independent environment. This is a proposed direction, not a result.

[Latest paper and build instructions](../paper/README.md) · [Raw analyses](../evidence/) · [Public browser demo](https://kw2828.github.io/OpenJev/)
