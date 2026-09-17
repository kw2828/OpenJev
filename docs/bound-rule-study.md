# Entity-bound recurrent rule operators

**Result: explicit binding plus recurrence solves these controlled-English questions. The fixed logical solver also solves them.** All 12 learned fits and three fixed controls completed. The [previous text comparison](rule-crossencoder-study.md) scored near chance after controlling for a question-negation shortcut. This experiment tests explicit entity binding and repeated rule application on previously unused development worlds.

![Bound-rule operator results](../evidence/bound-rules-v1/development.png)

| Condition | Same-depth macro accuracy | Shift macro accuracy | Counterfactual pair success |
| --- | ---: | ---: | ---: |
| Facts only | 67.44% | 50.00% | 0.00% |
| Fixed one step | 84.92% | 50.00% | 0.00% |
| Learned one step | 86.61% | 51.40% | 0.00% |
| Collapsed entities, 16 steps | 61.75% | 56.67% | 0.00% |
| Learned six steps | 100.00% | 100.00% | 66.67% |
| Learned sixteen steps | 100.00% | 100.00% | 100.00% |
| Fixed solver, 32 steps | 100.00% | 100.00% | 100.00% |

Both recurrent variants answered all 2,941 same-depth and 949 shifted development questions correctly in every seed. Sixteen steps also answered all 960 constructed challenge questions correctly in every seed. Six steps failed longer support chains, despite perfect RuleTaker development accuracy. Merging entity identities reduced shift macro accuracy to 56.67%; the one-step model reached 51.40%. This isolates a useful role for binding and recurrence within the supplied grammar and logical operations.

The **frozen selector chose six steps**: both recurrent candidates tied on the primary metric, and six appeared first in the predeclared ordering. Its 66.67% pair success failed the required 95%, so the recorded continuation gate remains failed. We do not silently substitute the sixteen-step candidate to turn that gate into a pass. The sixteen-step result is nevertheless preserved as a successful tested condition and is available for a local demonstration. It is not an independent confirmation or a win over the fixed solver.

Training required 15,360 optimizer updates and 50.53 summed CPU seconds. Parsing and grounding the two entity conditions and challenges took 0.86 seconds. The handwritten parser covered all 2,450 source worlds; no unsupported world was dropped. Timing excludes final evaluation, checkpoint writing and plotting. A runtime advantage over a conventional solver has not been measured.

[All fits and frozen selection](../evidence/bound-rules-v1/summary.json) · [Execution receipt](../evidence/bound-rules-v1/receipt.json) · [Fresh-world data manifest](../evidence/bound-rules-v1/data-manifest.json) · [All 12 compact weight files](../evidence/bound-rules-v1/weights/)

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

Try the sixteen-step model without downloading an encoder. This demonstration uses the first predefined seed, 17, and fixed true/false candidates:

```sh
uv run --extra train python scripts/bound_rule_demo.py examples/bound-rules.json
```

It returns caller-owned question IDs, a choice and both candidate scores. The example concludes that Mira is kind and Nemi is not. Reversing `Mira visits Nemi` to `Nemi visits Mira` changes the first conclusion. The parser supports this controlled grammar; unsupported syntax raises an error. Existing Qwen and Doom demos are separate models.

To reproduce the study:

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

## Next mechanism to test

The demonstrated weakness is now computation depth: six steps can be confidently wrong on an eight-step chain. A confidence-only gate may stop before a relevant fact reaches the query. The next comparison should freeze an adaptive rule-propagation policy against fixed budgets, entropy gating, a query-dependency residual heuristic, and a conventional solver that stops when settled. Charge the gate, parsing and graph analysis to total runtime. Dependency analysis and fixed-point stopping are established techniques; any novelty claim requires a distinct mechanism and stronger evidence than these perfect grammar-bound scores.
