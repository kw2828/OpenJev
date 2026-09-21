# Olfactory search: a candidate for compact recurrent belief models

21 September 2026. Source review only. No environment import, simulator run,
dependency installation, weight download or training has occurred. This is a
candidate for a later experiment, not an admitted learning study or a new result.

The [OTTO benchmark](https://github.com/auroreloisy/otto-benchmark) accompanies
Loisy and Heinonen's *Deep reinforcement learning for the olfactory search
POMDP: a quantitative benchmark* (2023). It includes isotropic and windy cases,
classical information-seeking policies, neural value policies and interfaces
for Sarsop/Perseus policies. Its task is directly relevant to insect navigation:
movement changes the future distribution of odor detections while an agent
searches for a hidden source. Biological relevance alone establishes no benefit
from a connectome.

The inspected upstream commit is
[`a6aaef6507cffd2aff79291c1019f506f616bbef`](https://github.com/auroreloisy/otto-benchmark/tree/a6aaef6507cffd2aff79291c1019f506f616bbef).
The tracked clone is unchanged. [Source inventory and hashes](../output/otto-source-review-v1/receipt.json)
bind the files inspected. The repository uses the MIT license.

## Important interface distinctions

- The [Gym wrapper](https://github.com/auroreloisy/otto-benchmark/blob/a6aaef6507cffd2aff79291c1019f506f616bbef/isotropic/classes/gymwrapper.py)
  supplies an agent-centered **Bayesian belief grid**, not just the current odor
  detection. That observation already summarizes history. Adding recurrence to
  this route would not itself test memory from raw observations.
- The [source-tracking class](https://github.com/auroreloisy/otto-benchmark/blob/a6aaef6507cffd2aff79291c1019f506f616bbef/isotropic/classes/sourcetracking.py)
  exposes movement, hit counts, detection likelihoods and belief updates. It has
  both a sampled-source mode and a belief-based mode. Their terminal quantities
  and evaluation semantics differ; do not mix their search-time measurements.
- The [neural policy](https://github.com/auroreloisy/otto-benchmark/blob/a6aaef6507cffd2aff79291c1019f506f616bbef/isotropic/classes/rlpolicy.py)
  evaluates next-step beliefs after all candidate actions and hit outcomes.
  It uses an explicit Bayesian update in addition to its neural value model.
  Its computation and information cannot be equated with a raw-hit recurrent
  policy by comparing network parameter counts alone.
- Source and hit draws repeatedly construct `np.random.RandomState()` without
  a seed. Setting NumPy's global seed does not control those fresh generators.
  A reproducible paired comparison needs an explicit, qualified random-stream
  integration; a seed list in a manifest is insufficient.
- The initial belief is already conditioned on an initial hit. A raw-observation
  actor needs that same hit; withholding it would create unequal information.
  Non-found visited cells are also eliminated, including after a zero hit.
  The filter normalizes only when total mass exceeds `1e-10`, so qualification
  must check finite normalized beliefs rather than assume numerical exactness.
- The benchmark pins TensorFlow 2.8 and a protobuf release candidate. The Gym
  wrapper also imports Gym, absent from the main dependency list. A local
  isolated runtime and actual reset/step qualification remain to be established.
  The pure simulator and heuristic sources are not proof that the full released
  neural-policy stack works in the current environment.

## A useful experiment, if the task qualifies

First reproduce a competent public Bayesian/classical controller under a fixed
observation and termination contract. Verify initialization, boundaries, hit
likelihoods, random draws and complete search-time accounting before scoring a
model. Keep the official configurations and original results separate from any
adapted execution path.

For a later recurrent comparison, give learned actors only public movement,
hit counts and boundary information. A reconstructible Bayesian filter can be
a reference with the same supplied likelihood rules; its full belief grid must
not silently enter the learned actor. Compare recent histories, a compact
finite-state controller, a GRU and a structured action-conditioned belief model.
Measure success/search time against total inference and planning cost, including
belief construction. Evaluate on untouched episodes and a predeclared parameter
shift. A memory-reset penalty by itself is insufficient if a simpler controller
achieves the same outcome.

Only after useful learned state is established should biological wiring compete
with matched rewired graphs, the same update rule and the same state budget.
Fast local plasticity, latent prediction and a new reward objective are separate
mechanisms. Do not bundle them and attribute a gain to the connectome.

This is a more direct biological-control candidate than chess, but superiority,
runtime feasibility, required memory depth and a research contribution remain
unproven. The immediate deliverable would be a qualified reproducible interface
and a competent classical reference, not another speculative architecture claim.
