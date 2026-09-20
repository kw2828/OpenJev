# Queryable beliefs: contribution and test boundary

Primary-source review, September 20, 2026. No new experiment, model call or
inspection of the running alignment campaign's predictions was performed for
this note. The [nine-fit comparison](dialogue-token-alignment-scientific-status.md)
must finish before its result informs another architecture choice.

## What recent work already covers

| Primary source | Established mechanism | Consequence for OpenJev |
|---|---|---|
| [Towards a Belief-Based World Model for LLM Agents, 2609.00455v1](https://arxiv.org/html/2609.00455v1) | Gives an LLM separate belief-query and simulation interfaces. The experiment uses hand-designed state spaces and belief updates; its ScienceWorld action-validity component uses the environment as an oracle. | A queryable candidate distribution is already prior art. Their interface result does not demonstrate learning a recurrent belief estimator. |
| [Belief Memory, 2605.05583v1](https://arxiv.org/html/2605.05583v1) | Retains alternative conclusions, merges support with Noisy-OR, and returns conclusions with their probabilities. Section 3.3 explicitly distinguishes these per-conclusion evidence values from a normalized posterior over mutually exclusive alternatives. | Keeping multiple hypotheses is already prior art. OpenJev must specify what its normalized, candidate-relative scores mean and cannot borrow a calibration claim. |
| [Optic-lobe eligibility-trace study, Scientific Reports 2026](https://www.nature.com/articles/s41598-026-52140-3) | Tests connectome-constrained dynamics, information/cost tradeoffs and local eligibility-trace learning against matched nulls and alternative learning rules. Energy is a modeled proxy; part of the learning evaluation measures alignment to biological wiring. | This motivates explicit learning-rule and topology controls. It does not establish better text decisions, chess strength or robot control from fly wiring. |

The first paper establishes a reason to expose uncertainty to a policy. The
second supplies a concrete probabilistic-memory comparator. Neither allows a
claim that another normalized recurrent output is new. The connectome study
also makes a cost-sensitive plasticity proposal less distinctive unless its
mechanism and task evidence go beyond those comparisons.

## A sharper question, conditional on better observations

Our hypothesis is that a small recurrent state could learn to discount repeated
or correlated support while still responding promptly to an actual state
revision. That would address an observable failure mode rather than add a
biological mask to a classifier. This review does **not** establish that the
mechanism is novel, needed in the current dataset, or effective.

The test must distinguish a repeated report of one event from new independent
evidence and from a genuine change. Identical observations do not always imply
duplication, so a duplicate-invariance requirement needs an explicit event or
source identity. Use only identities available to every deployed comparator;
do not supply hidden causal labels to the proposed model.

The existing [competitive-evidence design](dialogue-evidence-design.md) already
identifies repeated lexical history as an overconfidence risk. This reading
adds stronger external comparators and probability-semantics boundaries; it
does not turn that older product-update proposal into a new architecture.

Necessary controls include latest-value overwrite, explicit event deduplication,
a scalar forgetting filter, a conventional GRU, and the relevant delta-rule
memory. Give all arms the same observation model, history and source metadata.
If bookkeeping explains the gain, credit that. If a better encoder explains it,
credit representation learning. Learned state must improve decisions after
revision and subsequent retention, on common full-trajectory denominators.

This would initially be a memory-estimation experiment. Calling it a world
model would additionally require an action-conditioned transition that improves
planning or control. Calling it a connectome contribution would require an
interaction with biological topology against matched rewires. Neither claim
follows from the current gold-previous-state observation diagnostic.

## Choose the next work from the completed comparison

| Completed finding | Consequence |
|---|---|
| Alignment passes the fixed behavioral rule against both controls | Carry that observation component into the stronger autonomous-memory comparison. The alignment adaptation itself remains established prior art. |
| Token mean accounts for any useful improvement and alignment fails its rule | Prefer the simpler component where supported; do not credit alignment. Measure any memory contribution separately. |
| Neither new arm yields a useful decision improvement | Revisit semantic representation and supervision, including the small fine-tuned encoder control, before adding another recurrence. |
| The run does not complete and authenticate all nine fits | Preserve the incomplete attempt and do not select a method from partial quality results. |

The [evidence review](dialogue-recurrent-evidence-review.md) defines the stronger
comparators and freshness limits. This decision table changes no current
training recipe, threshold or evaluation access.
