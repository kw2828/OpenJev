# Next direction for the finite recurrent model

**Prospective design only, not registered or executed.** The design below was prepared before inspecting the replication's result metrics and reserves no seeds. The previously closed longer-horizon training follow-up remains closed.

The completed [five-seed replication](finite-reuse-replication-results.md) subsequently failed its continuation rule, with 45/54 conditions passing. The next comparison would be a separate diagnosis of initialization dependence, not promotion of that failed candidate or a claim that the noise-shift failure is explained.

## First isolate dependence on the readout initialization

Keep the current world and training recipe. Removing the head prior and introducing nonuniform transitions together would leave a failure ambiguous between readout learning, transport misspecification and their interaction.

Use three arms across five fresh paired fit seeds, for 15 fits:

1. Rounded transport with the existing learned-head initialization, as an anchor.
2. Rounded transport with task-independent random head logits.
3. Matched-free transport with the identical task-independent random head logits.

Retain 352 parameters, the linear mass readout, public prefixes, H1/H2 supervision and 1,024 prefix plus 3,072 joint updates. Fix the random-head distribution before qualification, independently of evaluation results and the true state-to-cost map. Pair its draws across arms 2 and 3. Supervised cost labels remain.

Match free and rounded initial transition probabilities, emissions, hazards and random head. Pair batches and report actual compute; initial-function matching does not match gradients. The two rounded arms must have identical dynamic parameters at the prefix-stage boundary because their unchanged prefix objective and optimizer exclude the head.

Head-column shuffling is insufficient. A consistent latent permutation leaves the function unchanged. Shuffling columns against exchangeable random dynamics retains the known cost-template shape, amplitude and two-columns-per-action structure. Generic random logits remove that prior but also change optimization scale; describe the full initialization intervention honestly.

Require the existing absolute SHORT, BLIND and OBSERVED criteria and paired H4/H8 regret requirements against random-head matched-free, including the mean reduction requirement. The anchor cannot rescue failure. SHORT failure stops continuation without establishing a long-horizon transport defect. Passing the absolute criteria but failing the matched comparison leaves the transport advantage without the head prior unsupported. No extra updates or initialization search follow.

## Later, test a learned common stationary distribution

A separate hypothesis would learn positive mass \(\pi\) and asymmetric action flows \(F_a\) whose row and column marginals both equal \(\pi\), then set

\[
T_a=F_a\operatorname{diag}(\pi)^{-1}.
\]

Each column-stochastic transition preserves \(\pi\), without symmetry or detailed balance. Keep reset mass separate; the hazard-modified surviving operator need not preserve \(\pi\).

Use a separate registration with actual nonuniform transition laws and task-independent heads throughout. Changing emissions or reset frequencies alone does not challenge uniform stationarity. Common stationarity remains a prior whose limits require an environment without it.

The four eight-state transition manifolds have 196 effective dimensions with fixed uniform mass, 203 with learned common mass, and 224 for unrestricted columns. These are ideal mathematical dimensions, not measured finite-algorithm ranks. Report useful parameters and compute without inert padding. Separate stationary mass for each action largely removes the shared structural hypothesis.

## Primary prior art and contribution limits

- [Bengio and Frasconi, *An Input Output HMM Architecture* (1994)](https://proceedings.neurips.cc/paper/1994/file/8065d07da4a77621450aa84fee5656d9-Paper.pdf) establishes input-conditioned probabilistic recurrence and likelihood-based training. Nonnegative recurrent filtering is established methodology.
- [Downey et al., *Predictive State Recurrent Neural Networks* (2017)](https://proceedings.neurips.cc/paper/2017/file/2bb0502c80b7432eee4c5847a5fd077b-Paper.pdf) connects bilinear observation-gated updates, normalization, predictive states and BPTT. Sharing filtering and prediction mechanisms is not itself novel.
- [Mardt et al., *Deep Learning Markov and Koopman Models with Physical Constraints* (2020)](https://proceedings.mlr.press/v107/mardt20a.html) learns latent dynamics and equilibrium reweighting with optional stochasticity and reversibility constraints. This is direct prior art for learned stationary structure.
- [Altschuler, Weed and Rigollet, *Near-linear Time Approximation Algorithms for Optimal Transport via Sinkhorn Iteration* (2017)](https://proceedings.neurips.cc/paper/2017/file/491442df5f88c6aa018e86dac21d3606-Paper.pdf) provides matrix scaling and rounding to prescribed marginals. Those numerical ingredients do not establish recurrent-learning performance.
- [Liu, Klinger and Rotskoff, *Optimal Parameterization of Nonequilibrium Generalized Master Equations from Discrete-time Experimental Data* (2026 preprint)](https://arxiv.org/abs/2606.28289) represents stationary flux as transport from an estimated stationary distribution to itself. Its coarse-grained, stationary-vector-conditioned estimation differs from jointly learning our hidden action-conditioned filter, but the stationary-flow construction is already prior art.

Any possible contribution must come from the specific architecture, its analysis and reproducible evidence beyond favorable synthetic assumptions. These ingredients alone establish neither architectural novelty nor biological relevance.
