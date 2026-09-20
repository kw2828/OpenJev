# What selective retention can change in one step

Source analysis, not a new training or evaluation result. This calculation was
independently reviewed against the existing update implementation. It explains
a distinction between two current model families and narrows the next diagnostic;
it does not establish a new architecture or explain the empirical failures.

## An exact matched-factor comparison

Let `b` be the current normalized candidate belief, `w` the normalized writer
distribution, and `r` the candidate departure probabilities in `[0,1]`. All sums
are over valid candidates. Hold all three identical across the two updates and
define `m = sum_i b_i r_i`. The [implemented transition](../src/openjev/research/dialogue_copy_memory.py)
has the following real-arithmetic form:

```text
scalar:     p_i = (1-m) b_i + m w_i
selective:  q_i = (1-r_i) b_i + m w_i
difference: q_i - p_i = b_i (m-r_i)
```

The shared write term cancels. The total variation distance therefore is

```text
TV(p,q) = 0.5 sum_i b_i |r_i-m|
        <= 0.5 sqrt(Var_b(r))
        <= (max r - min r) / 4
        <= 1/4.
```

The first inequality is Cauchy-Schwarz. For the second, write `a=min r` and
`d=max r`. Since `(r-a)(d-r) >= 0`, taking the belief-weighted expectation gives
`Var_b(r) <= (m-a)(d-m) <= (d-a)^2/4`.

There is also a useful certainty-dependent bound. Choose `j` with maximal
belief. The triangle inequality and `|m-r_j| <= E_b|r-r_j|` give

```text
TV(p,q) <= E_b|r-r_j| <= (1-max b) (max r-min r).
```

Thus the matched-factor updates are identical for a one-hot belief, or for
departure probabilities constant on the belief's support. At `max b=0.99`,
their one-step TV distance is at most `0.01`. This is a probability-distance
bound, not a one-percentage-point accuracy bound: small changes can flip an
argmax near a tie.

The global `1/4` bound is tight for `b=(1/2,1/2)` and `r=(0,1)`, for any shared
writer. Finite sigmoid logits approach those endpoints. A finite-logit example
is `b=w=(1/2,1/2)` and `r=(1/4,3/4)`: scalar returns `(1/2,1/2)`, selective
returns `(5/8,3/8)`, and TV is `1/8`, exactly one quarter of the gate range.

## What this does and does not identify

Selective retention has appreciable additional one-step effect only when both
the prior belief is distributed and the departure gates differ. Entropy alone
does not establish that the extra mechanism is being used. The initializer sets
all departure logits to `-3`, so paired scalar/selective models initially use
the same transition outputs even for diffuse beliefs. Equality there does not
imply equal derivatives with respect to candidate departure logits. Training
can subsequently make their factors and trajectories differ.

Either update can still make a large corrective write through `m*w`. A nearly
one-hot stale belief does not make correction impossible. The [corrected V2
results](dialogue-copy-v2-results.md) cannot be explained by this bound alone:
their separately trained models need not have matched `b`, `r` or `w`, and small
one-step differences can accumulate. This is not a bound on accuracy, NLL,
learning gradients, separately trained models, or full trajectories.

The proof assumes exact normalized probabilities. [V2](../src/openjev/research/dialogue_copy_memory_v2.py)
normalizes in floating-point arithmetic within monitored tolerances. The
calculation neither replaces those monitors nor validates the old unnormalized
V1 states. The source hashes used here are:

| Source | SHA-256 |
|---|---|
| `dialogue_copy_memory.py` | `182c71a944c6d787533bf129bca1631ff3b366b8559e3db7567e7fd9b19a8717` |
| `dialogue_copy_memory_v2.py` | `3fc1e84e5fe9da0076e8d83e7d5c67e61607d67b49ce2b0357a81e2206d3714b` |

## Consequence for the next experiment

Resolve the unfinished observation comparison before adding a larger recurrent
state. The [token study](dialogue-token-results.md) timed out without a complete
quality result; its pending [execution optimization](dialogue-token-shared-columns-status.md)
cannot answer that question. The matched slot/candidate-attention by
readout/scalar comparison remains the required observation control. Readout
already receives deterministic literal-history features, so the comparison
measures learned state beyond that buffer, not memory versus no memory.

For any later prospective selective-memory experiment, record the exact
matched-factor TV above during evaluation, alongside prior concentration and
gate dispersion. Compute its counterfactual using that model's own factors;
do not label it a comparison against a separately trained scalar model. Include
all valid updates and account for diagnostic cost. This requires a new declared
evaluation; the closed V2 and incomplete token runs are not being replayed.

Pair that diagnostic with revision followed by delayed-retention endpoints on
the same dialogues. Report joint correctness and both unconditional endpoint
accuracies for every method on a common denominator. Conditioning retention on
each model's own successful revisions would select different examples. Exact
extraction rules, support, dialogue weighting and held-out use must be frozen
before a future evaluation; no endpoint panel was extracted here.

The [competitive-evidence proposal](dialogue-evidence-design.md) remains
conditional. Its stronger scalar control should receive the same writer
competition features as the proposed product update. If that scalar accounts
for any gain, credit the added evidence access rather than the update law.

These distinctions also limit literature borrowing. Gated DeltaNet combines
decay with delta-rule writes in associative memory; that is a related baseline,
not evidence that another residual gate will repair this task. [Primary paper](https://arxiv.org/abs/2412.06464v3).
Prior neural belief-tracking work already studies uncertainty features and
their effects on downstream dialogue policies. A normalized candidate
distribution alone establishes neither calibration nor policy benefit.
[Primary paper](https://aclanthology.org/2021.emnlp-main.623/).
V-JEPA 2-AC instead learns action-conditioned latent prediction for robotic
planning. Passive categorical dialogue tracking does not test that capability.
[Primary paper](https://arxiv.org/abs/2506.09985).
