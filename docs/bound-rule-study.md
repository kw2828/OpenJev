# Entity-bound recurrent rule operators

**Status: frozen before training.** The [previous text comparison](rule-crossencoder-study.md) scored near chance after controlling for a question-negation shortcut. This experiment tests explicit entity binding and repeated rule application on previously unused development worlds.

The model parses only the English context and questions. A handwritten parser supports the controlled RuleTaker grammar, including unary properties, directed relations, conjunctions, variables and negation. It grounds variable rules over named entities. It cannot be presented as general English understanding. Dataset proof strings and supplied logical forms are never model inputs.

The parser plus fixed logical operator reproduced all 19,809 training answers before the experiment was frozen. That is a parser/semantics audit on training data, not learned efficacy or independent generalization.

## What is learned

A 225-parameter MLP learns how antecedent truth values combine into a rule activation. Inputs are sorted literal values, padded with ones, and the number of real antecedents. The same operator is reused at every rule, entity and recurrent step. It starts randomly and receives only final question labels, with no local truth-table or proof supervision.

Binding, `1-p` negation, max-OR aggregation and multiplicative inhibition are supplied. These are substantial inductive biases. The state is recomputed synchronously from facts and rule activations, allowing negative heads to inhibit positive conclusions. The fixed comparator replaces the learned conjunction with minimum and performs 32 iterations. Cyclic or slowly settling probabilities are counted through the last-step change; fixed iteration limits are not a general proof of convergence.

[Differentiable proving](https://arxiv.org/abs/1705.11040), [Neural Logic Machines](https://arxiv.org/abs/1904.11694), and [ProbLog inhibition](https://dtai.cs.kuleuven.be/problog/tutorial/basic/10_inhibitioneffects.html) are existing prior work. This experiment is not a new reasoning algorithm or a reproduction of those models. A neural operator matching an ordinary solver would be a working foundation, not an ICLR novelty result.

## Frozen comparison

| Condition | Operator | Iterations | Entity binding |
| --- | --- | ---: | --- |
| Facts only | No rule application | 0 | Explicit |
| Fixed one step | Minimum conjunction | 1 | Explicit |
| Fixed solver | Minimum conjunction | 32 | Explicit |
| Learned one step | Shared MLP | 1 | Explicit |
| Learned six steps | Shared MLP | 6 | Explicit |
| Learned sixteen steps | Shared MLP | 16 | Explicit |
| Collapsed entities | Shared MLP | 16 | All names merged |

The four learned conditions each use seeds 17, 29 and 43, for 12 fits. They share 2,000 training worlds and 19,809 questions, 40 epochs, batches of 64 worlds, AdamW learning rate 0.01 and weight decay 0.0001, gradient norm cap one, and identical world shuffles per seed. Only final-epoch checkpoints are evaluated. All learned conditions have the same 225 parameters; more recurrence costs more computation.

The new development packet excludes every world previously selected for training or evaluation. It contains 300 same-depth worlds and 150 depth-3-5 worlds. Training worlds are unchanged. Question labels remain verified against the source Problog release. Selection uses four-group macro accuracy, weighting gold label crossed with question negation equally. Raw accuracy and probability scores are also reported. No official RuleTaker test members or CLINC confirmation scores are read.

## Counterfactual and invariance checks

The predeclared generator creates 240 context pairs across chain lengths 3-8. Each pair reverses the subject and object of a directed relation while preserving the word multiset and question. That reversal changes whether a support or inhibition rule applies. Positive and negated questions give 960 answers in 480 worlds, balanced across all four label/negation groups. An additional version inserts facts about an irrelevant entity.

Pair success requires all four answers in a pair to be correct. Invariance compares predictions before and after irrelevant facts. Analytic labels follow the construction and are checked against the fixed operator. These are constructed mechanism checks sharing the parser grammar, not a separate natural-language benchmark. Parser-enforced renaming and body-order invariance are implementation properties, not learned accomplishments.

Select six or sixteen recurrent steps by shift macro accuracy. The functional continuation rule requires at least 95% macro accuracy, at most a one-point gap behind the fixed solver, at least ten points over the collapsed-entity control, and at least 95% counterfactual-pair success. Passing supports further research, not a claim of novelty or an advantage over the fixed solver.

## Reproduce

```sh
.venv/bin/python scripts/prepare_bound_rules.py \
  --archive runs/ruletaker-source/rule-reasoning-dataset-V2020.2.5.zip \
  --previous runs/ruletaker-depth-v1/packet --out runs/bound-rules-v1/packet
.venv/bin/python scripts/bound_rule_study.py freeze \
  --packet runs/bound-rules-v1/packet --out runs/bound-rules-v1/plan.json
.venv/bin/python scripts/bound_rule_study.py run \
  --packet runs/bound-rules-v1/packet --plan runs/bound-rules-v1/plan.json \
  --out runs/bound-rules-v1/fits
.venv/bin/python scripts/bound_rule_study.py report \
  --packet runs/bound-rules-v1/packet --plan runs/bound-rules-v1/plan.json \
  --runs runs/bound-rules-v1/fits --out runs/bound-rules-v1/summary.json
```

Code, packet, plan and checkpoint hashes are checked. Raw source worlds and per-question probabilities stay local. Outputs are bounded truth probabilities under supplied logical structure, not calibrated confidence in arbitrary English claims. Astra supervision has not run.
