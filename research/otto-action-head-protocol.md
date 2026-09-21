# Direct actions from recurrent evidence: prospective learned pilot

21 September 2026. The [autonomous fixed-memory comparison](otto-spectral-control-results.md) preserved near-full-Bayes utility with 80.79% less evolving state, but added 31-39% controller computation and failed both decision rules. This independently justified pilot trains a readout that avoids dense decoding and lookahead. It does not revise those failures or claim that distillation, compact memory or recurrence is new.

## Scientific question and existing work

Does a compact evidence accumulator retain useful action information when all representations receive the same supervised policy readout, and does the resulting autonomous controller reduce complete computation without losing search utility?

[OTTO's deep RL baseline](https://arxiv.org/html/2302.00706v2) learns a value of full belief and still evaluates successor beliefs. [Policy distillation](https://arxiv.org/html/1511.06295v2) already trains smaller action networks from teacher scores. [Recurrent predictive-state policy networks](https://proceedings.mlr.press/v80/hefny18a.html) already couple a recursive state estimator to a reactive policy. [Value-directed POMDP compression](https://papers.neurips.cc/paper/2192-value-directed-compression-of-pomdps.pdf) already emphasizes preserving decisions rather than reconstruction alone. A positive pilot would justify a narrower architecture experiment; it would not establish a new algorithm or paper-level contribution.

The essential learned comparator is full belief with a fast head. If that explains the gain, the result supports planner distillation. An unsuccessful full-belief head also prevents attributing candidate failure to compact memory alone.

## Four representation families and twelve fits

Keep the published evidence-update implementations, numerical kernels, masks and public observation API unchanged. Train each family with fitting seeds **7901, 7902, 7903**:

- `dct16_neutral`: 256 additive DCT coefficients, neutral finite extension, all historical hard exclusions.
- `dct16_nearest`: the same rank and mask, distance-one finite extension.
- `recent32_hard`: 32 chronological readings, all historical hard exclusions, initial prior retained by the original actor.
- `full_bayes`: the complete public-history posterior and hard exclusions.

Each head is an MLP with hidden widths **64 and 32**, Tanh activations and four lower-is-better action scores. DCT/recent heads have **3,080 inputs and 199,396 parameters**. The full-belief head has **5,633 inputs and 362,788 parameters**. Full belief is deliberately a stronger control with more active parameters. Equal hidden widths are not described as equal parameter capacity.

DCT input is the elementwise signed `log1p(abs(coefficient))` of all 256 coefficients, without inverse transformation, dense reconstruction or planning. Recent input contains 32 right-aligned slots, each with normalized absolute coordinates, four hit indicators, validity and relative age. Full input contains all **53*sqrt(probability)** values, an invertible scale with grid RMS one. Every family also receives its entire **2,809-bit support mask** and 15 public context features: normalized position, initial-hit category, current-hit category, normalized log elapsed step, legal-action indicators and the supplied sensing length divided by four.

Fit feature means and population standard deviations on TRAIN only, with each standard deviation floored at one. No validation or evaluation value fits preprocessing. No feature clipping, learned observation model, support repair, teacher fallback or hidden posterior is added to candidates. The initial prior's identity is provided by initial hit and sensing length; candidates do not receive a dense prior feature vector.

The primary comparison includes the strong recent32_hard head because DCT also retains old hard exclusions. A recent32 control without those exclusions is not part of this pilot.

## Fresh shared training and validation trajectories

Collect **384 TRAIN episodes** with seeds 670001-670384 and **96 VALIDATION episodes** with seeds 680001-680096, all at baseline sensing length three. Initial hit is `1 + case_index % 3`, giving balanced conditional strata. Each trajectory lasts until source finding or **256 moves**. Capped training trajectories are retained.

The behavior policy chooses the full-Bayes space-aware teacher action with probability 0.85 and a uniform legal action with probability 0.15. Exploration uses a separate local PCG stream keyed by `[episode_seed,99]`. Record the exploration branch, teacher scores, chosen action and public transition. The four representations evolve on exactly the same public stream. The teacher is reconstructed from public history and checked against native filtering; hidden source locations never enter features or supervision.

At each pre-action prefix, distill the same four full-Bayes heuristic scores to all four families. These scores are not Q-values or calibrated probabilities. On valid actions, subtract the minimum score, divide by `max(maximum-minimum,1e-8)`, negate and divide by the fixed temperature **0.25**, then normalize with softmax. Illegal actions have target mass zero. This relative within-position target preserves ranking, not the teacher's absolute objective scale.

All decision rows receive equal training weight. The resulting state distribution favors long trajectories and differs from mixture-weighted episode evaluation; this is disclosed rather than called distribution matching. No teacher targets from an evaluated cohort are added back to training.

## Fixed optimization and checkpoint selection

For every fit use CPU float32, one numerical thread, deterministic Torch operations, Adam learning rate **0.0003**, batch size **256**, **40 epochs**, and gradient-norm clipping at five. Use negative model scores as categorical logits and mask illegal actions. Optimize cross entropy against the fixed soft teacher target. Fitting-seed permutations are identical across families. Do not change hyperparameters, sample weights or training length after observing losses.

Evaluate validation cross entropy every fifth epoch. Select the checkpoint with minimum finite validation loss; the first checkpoint wins an exact tie. Preserve all eight validation records per fit, every selected epoch, counts, losses, weights and standardizers. Selection never sees autonomous evaluation outcomes. Confirm NumPy-export inference against Torch within 2e-5 on eight validation feature rows, then serialize, reload and use the selected checkpoint for autonomous evaluation. This numerical check does not qualify search performance.

The sensing-length feature is constant during TRAIN. After standardization it is zero, so its input weights receive no learning signal; shifted evaluation activates an untrained input direction. Other rare or constant spatial features may similarly extrapolate. Thus lambda4 results are a combined prior/readout-transfer test, not evidence of learning the new sensor model. The full-belief control supplies the exact changed prior implicitly and helps expose this limitation. The longer evaluation horizon also tests beyond the 256-step training support.

## Autonomous comparison

Keep the 53x53 grid, four hit categories, Euclidean sensing, R_dt=2 and **2,188-move horizon**. Evaluate sensing lengths three and four with their separately supplied kernels and priors. This is the same fixed-grid known-model shift as the previous study, not the automatically sized upstream benchmark or unknown-model identification.

Each regime has **48 fresh cases**, eight blocks with two cases per initial-hit category. For case `c=0..47`, use block `c//6`, initial hit `1+(c%6)//2`, baseline seed `690001+c` and shifted seed `700001+c`. Run all twelve learned fits and four unchanged analytic planners (full Bayes, both DCT fills, recent32_hard): **1,536 episodes**. Rotate the sixteen-arm order by `(regime_index*48+case)%16`.

All arms share source/observation random channels within a case. Different paths generally produce different observations. Every actor resets at each episode. Public packets contain only position, hit, terminal flag, elapsed step and legal actions; seed IDs, source, simulator beliefs and draws are evaluator-only. Use the original first-within-1e-10 score tie rule. Update every nonterminal reading, including the final horizon-censored reading; never encode the found sentinel. There is no teacher correction, fallback, outcome-based stopping or replacement episode.

Report all fits, success counts, per-hit strata and all eight blocks separately under both regimes. Unsuccessful searches contribute all 2,188 moves. Compute each conditional-stratum mean first, then weight by that regime's qualified initial-hit mixture. Family means average the three fitting seeds; do not pick a favorable seed or fill. Validation agreement alone cannot establish autonomous competence.

## Prospective interpretation and continuation rule

Full-belief readout competence requires **each fit** to attain at least 95% weighted success and mean capped moves at most 105% of the full-Bayes analytic planner, in **both regimes**. A failed prerequisite is evidence against the readout/data recipe; it does not isolate a memory mechanism.

For each DCT fill in each regime, require all ten conditions:

1. Every fit has weighted success at least 95%.
2. Family-mean success is no lower than the full planner.
3. Family-mean capped moves are at most 105% of the full planner.
4. Family-mean complete controller cost is at most 80% of the full planner.
5. Every fit costs less controller computation than the full planner.
6. Family-mean moves are at most 95% of the recent32_hard head family.
7. At least six of eight paired blocks improve versus the recent head family, averaging paired fit seeds within each block.
8. Family-mean moves are at most 105% of the full-belief head family.
9. Family-mean controller cost is no greater than the full-belief head family.
10. Evolving array state is at most 20% of the full planner.

The pilot passes only if the full-head competence prerequisite and all forty candidate conditions pass. These are prospective practical screening rules, not a statistical significance test or proof of novelty. Even a pass leaves the learned-architecture claim false pending a distinct mechanism, stronger controls and independent confirmation. Both earlier fixed-memory failure flags remain unchanged.

## Cost, resources and execution closure

Use the pinned OTTO source and public seeded adapter. Reuse the existing twenty native qualification checks with their qualification fixtures before any new trajectories. These reused fixtures qualify integration, not fresh scientific performance. Require all local/upstream source hashes, runtime versions, plan identity and supervisor identity before native imports/work.

Freeze all sources and tests in a committed plan before the first native call. Run once under the suspend-inclusive supervisor with limits **5,400 seconds**, **8 GiB RSS**, **2 GiB output** and **3,485,696 native calls** (480*256 + 1536*2188 + 2048). Preserve failures and partial work; no rerun or budget extension under this plan.

Validate checkpoint weights once when constructing an immutable bytes-backed runtime head; keep input and output checks on every decision. Time feature construction, standardization, NumPy head inference, legal-action selection, all evidence updates and actor initialization inside complete controller cost. Charge shared spectral-model construction per 48-case workload where applicable. Measure checkpoint loading/validation and immutable runtime construction and charge one load per 48-case learned workload in each regime, even though the process physically loads a checkpoint once. Report training/collection costs separately, along with evolving state, immutable head/standardizer bytes, actor priors, shared model/kernel tables, temporary dense transform arrays and total RSS. Logical workload allocations are not disjoint whole-process timings. Single-pass CPU timings are descriptive, not deployment latency claims.

Preserve the closed worker receipt and independent supervisor terminal, every dataset/checkpoint hash, per-fit histories, raw public trajectories, complete evaluation rows, compact categorical draw records (channel, index, uniform, selected category and CDF mass), selected sources for evaluator checks, and both failure flags. Completed status requires exit zero, absent worker process group, full expected work and no late-failure marker. A separate saved-output reader will recompute reporting and publication before any effectiveness claim. Any future replay must use case zero from both regimes, preselected here, with all sixteen arms and illustrative playback timing.
