# Corrective recurrence

The [learned-head comparison](learned-associative-study.md) reached 95.10% development accuracy with a linear metric head. Dense attractive recurrence worsened accuracy, NLL and cost. This screen tests whether subtractive or residual updates help instead, while keeping the learned parameter count and optimization recipe fixed.

Six heads run across three paired seeds, all with 68,481 parameters:

| Head | Update before normalization | Steps |
| --- | --- | ---: |
| Metric | Linear projected query | 0 |
| Feedforward | GELU projected query | 0 |
| Attractive | `0.5*q0 + 0.5*recall` | 2 |
| Inhibitory | `q0 - 0.25*recall` | 2 |
| Residual | `state + 0.25*(q0 - recall)` | 2 |
| Residual | `state + 0.25*(q0 - recall)` | 3 |

`recall` is the softmax-weighted mean of the learned class vectors given the current state, using beta 10. The original projected query `q0` is held fixed within each forward pass. All heads use the same original-embedding rejection gate. This is a finite differentiable architecture ablation trained by backpropagation. It is not local predictive-coding learning, an implicit equilibrium solver, or a claim of biological fidelity.

The unchanged optimizer and data-access functions from the frozen learned-head study are reused through an explicit adapter. The adapter hashes itself, the original trainer, and both architecture modules. It supplies the new model factory/protocol and verifies the actual 18 output files before writing the completion count. Prior sources, results and checkpoints remain unchanged.

The training recipe is identical to the previous screen: 30 epochs, 15,000 known-intent training examples, three paired seeds, same minibatch order, final-step cross entropy, same optimizer and learning rate. Selection reads only the same development set. It does not load the 5,410-example confirmation or 1,550-example calibration arrays.

Select the corrective variant with the best mean in-scope accuracy. It must beat the best control by at least 0.5 percentage points, improve or match balanced utility and NLL, win at least two paired seeds, and stay within three times the head matrix-operation count. Passing only permits a separately frozen confirmation. Repeated development selection across these screens increases optimism; no development improvement is an independent benchmark claim.

```sh
.venv/bin/python scripts/corrective_associative_study.py freeze \
  --packet runs/associative-text-v1/packet --out runs/corrective-associative-v1/plan.json
.venv/bin/python scripts/corrective_associative_study.py train \
  --packet runs/associative-text-v1/packet --plan runs/corrective-associative-v1/plan.json \
  --out runs/corrective-associative-v1/fits --device cpu
.venv/bin/python scripts/corrective_associative_study.py report \
  --packet runs/associative-text-v1/packet --plan runs/corrective-associative-v1/plan.json \
  --runs runs/corrective-associative-v1/fits --out runs/corrective-associative-v1/summary.json
```

Related work motivates the question, not a novelty claim: [predictive coding with local recurrent processing](https://arxiv.org/abs/1805.07526), [deep equilibrium models](https://arxiv.org/abs/1909.01377), and the [associative retrieval sources](associative-text-study.md#sources).
