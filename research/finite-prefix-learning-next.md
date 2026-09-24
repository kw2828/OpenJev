# Proposal: separate cost-readout alignment from observation prediction

**Proposal only, not registered or executed.** The [completed prefix-loss
study](finite-prefix-learning-results.md) improved event prediction but failed
all three criteria. Observed cost error worsened even as observed KL improved.
The next question is whether training the cost readout jointly with the filter
helps align the learned representation with decisions.

Fixed C is not an expressivity impossibility: the true world is exactly
representable with the current fixed C and appropriate reset/branch operators.
This proposal tests optimization and representation alignment under the recipe,
not proof that a fixed readout cannot work or that its latent state is wrong.

## One change with an initialization control

Use the same shared filter and coefficient-one prefix objective in three arms:

1. **Fixed exact C:** the current known centered cost readout.
2. **Fixed softened C:** hold `C_delta = (1-delta) * C` fixed, with **delta = 0.1**.
3. **Learned bounded C:** set `C[:,s] = 0.25 - softmax(L[:,s])`, with softmax
   over the four actions. Initialize `L = log(P_delta)`, where
   `P_delta = (1-delta) * onehot(g(s)) + delta/4` and g is the current known
   cost-preferred action for state s. This exactly matches the softened control
   up to float64 roundoff before learning.

Only learned versus fixed softened C isolates readout trainability. A gain
over that control alone could merely undo its initial softening, rather than
realign latent coordinates. The exact-C arm remains the privileged anchor. The sign is essential: costs require
**0.25 minus probability**, not probability minus 0.25. The learned head adds
32 logits, with 24 identifiable degrees of freedom, to the existing 1,088
parameters. Its columns sum to zero and lie within (-0.75, 0.25); finite logits
cannot attain the true boundary vertices exactly. No delta sweep is allowed.

Keep the linear readout of probability mass. It must preserve the identity
between expected conditional cost and cost read from the expected state;
absorbed zero state must produce zero cost. Do not apply a nonlinear readout
after averaging states. All arms retain a privileged, world-aligned initial
readout; this is not learning a decision task from unstructured text or pixels.

## Paired training and honest evaluation

Pair reset/branch initialization, all public attempts, labels and minibatch
orders across three fit seeds per arm. Use fresh TRAIN/DEV namespaces and the
existing all-attempt collector, terminal-event masks, global event/endpoint
denominators and coefficient-one objective. Retain the current 480 epochs,
batch size 64, Adam 0.003 and gradient clip 5. All nine final checkpoints must
precede DEV generation. No new initializer, warm start, factorized transition
family, loss-weight sweep or RL method enters this comparison.

Preserve all three absolute criteria and support minima from the
[prefix-loss protocol](finite-prefix-learning-protocol.md). Report every seed,
H4/H8 blind regret, blind and observed cost MSE, event KL, prefix NLL, storage
and actual training/inference cost. Report both learned-versus-softened and
learned-versus-exact comparisons. An H8 mean gain cannot hide a failed H4 cell
or worsening conditional costs. Lower prefix NLL cannot rescue the criteria.
There is no retrospective winner selection from the previous checkpoints.

Before registration, qualify identical initial softened/learned functions,
head-only gradient differences, probability/cost bounds, zero absorbed costs,
linear expectation consistency, oracle-free inputs and full work accounting.
Freeze the new sources, protocol, runtime, paths and one scientific attempt
before generating data. Previous studies and their failed criteria remain
closed. This proposal itself supplies no new result or execution admission.

A passing study would support training/readout alignment in this favorable
synthetic family, not identify the true latent state or establish calibration,
native transfer or novelty. [Value equivalence](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)
is established context for decision-useful model representations; this proposal
does not implement that paper's complete algorithm or inherit its guarantees.
