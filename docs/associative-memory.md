# Separating reactive state from an associative store

**Status: supervised learnability diagnostic completed; no associative-policy RL result yet.** The initial seven-action check did not learn a robust solution. A [separate controlled follow-up](associative-learnability.md) found that candidate-conditioned training lets the GRU learn four examples, while ordinary recurrence remains unreliable on longer routes. This is not a new fast-weight algorithm or a reproduction of Titans, Gated DeltaNet or Dreamer.

The existing controller compresses everything into one 64-dimensional GRU state. The candidate adds a separate 16 by 16 matrix that can retain associations across an episode. Its 256 state values are runtime memory, not additional persistent learned weights. The controller still acts from the same partial observation and previous action.

From the encoded observation, learn a unit-normalized key and query, a bounded value and a write strength. With matrix `A`, key `k`, value `v`, query `q` and scalar strength `beta`:

```text
A_next = A + beta * outer(v - A @ k, k)
read   = A_next @ q
policy_features = recurrent_state + project(read)
```

This is a rank-one delta-rule update. It changes the association addressed by the key. It does not implement an explicit whole-matrix decay, stochastic world model, planning step or gradient descent optimizer at deployment. The read affects the actor and critic; it does not feed back into the GRU. Episode resets clear both states. A diagnostic store-only reset leaves the GRU intact, then allows the current observation to be written normally.

## Controls that would make a result interpretable

| Arm | Added mechanism | Question |
| --- | --- | --- |
| GRU | None | Does added machinery improve on the original controller? |
| GRU + feedforward adapter | Similar learned parameter count, no extra temporal state | Is the improvement explained by capacity? |
| GRU + global writes | Associative store with a learned scalar write strength | Does a store or globally slower overwriting help? |
| GRU + selective writes | Same store, input-dependent write strength | Does choosing when to write add value? |

The global and selective versions begin with identical effective write strength, 0.1. The selective gate's input weights start at zero. Both allocate the same gate module; the global version freezes its input weights at zero and learns only the bias. Initial encoder, GRU, actor and value weights match the frozen predictive-PPO backbone at a fixed seed.

| Arm | Registered parameters | Trainable parameters | State values per episode |
| --- | ---: | ---: | ---: |
| GRU | 89,864 | 89,864 | 64 |
| Feedforward adapter | 94,185 | 94,185 | 64 |
| Global writes | 94,137 | 94,073 | 320 |
| Selective writes | 94,137 | 94,137 | 320 |

The controls are close in learned parameter count, not exactly identical. Twenty-seven focused tests cover initial equivalence, resets, delta updates, causal replay and gradients. Those checks establish implementation mechanics only.

For a future trained comparison, retain paired cue swaps, state resets, native-start transfer and longer corridors. Add a store-only reset and report measured runtime, the extra state size and every seed. Log write strengths against separately audited cue visibility without supplying visibility labels to the controller. Selective writes must beat the global-write and feedforward controls before attributing an improvement to selective retention.

If every controller remains weak, first resolve training and exploration. A larger store cannot be credited with solving a task that no baseline learned, and an isolated successful episode cannot establish a useful mechanism. No new training budget, success threshold or confirmation claim is established by this implementation note.

## Prior art

- [Fast weights, 2016](https://arxiv.org/abs/1610.06258): temporary associative weights alongside ordinary neural activity.
- [MERLIN, 2018](https://arxiv.org/abs/1803.10760): predictive representations, memory and policy learning in partially observed tasks.
- [Differentiable plasticity, 2018](https://arxiv.org/abs/1804.02464): learned plasticity with reinforcement-learning experiments.
- [Fast-weight programmers and the delta rule, 2021](https://arxiv.org/abs/2102.11174): direct overlap with the update and input-dependent write strength above.
- [Recurrent fast-weight programmers, 2021](https://arxiv.org/abs/2106.06295): recurrent fast-weight controllers evaluated in reinforcement learning.
- [Gated DeltaNet, ICLR 2025](https://arxiv.org/abs/2412.06464): combines targeted delta updates with adaptive memory erasure. Our small controller does not implement its complete layer or training system.
- [Titans, 2025](https://arxiv.org/abs/2501.00663) and [MIRAS, 2025](https://arxiv.org/abs/2504.13173): broader learned-memory, retention and online-optimization frameworks.

A future contribution would need a more specific mechanism and evidence beyond these established ideas. Stronger performance on this small memory task would be an engineering result first.
