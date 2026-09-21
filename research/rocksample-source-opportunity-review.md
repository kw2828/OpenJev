# RockSample: source opportunity review

21 September 2026. **Conditional task candidate, not an admitted training study.**
The ordinary observation has real history aliasing and actions affect both future
information and reward. However, current upstream factory routes expose different
information under the same `perfect_memory` flag, and a compact exact belief
filter needs map information absent from the ordinary observation. Those issues
must be resolved before comparing architectures.

This review cloned and read source only. No dependency installation, imported
benchmark code, compilation, tests, simulator episodes, model calls or training
occurred. The failed OpenJev pooling recipe remains closed; this note supplies no
new performance result or world-model claim.

## Source identity

Official repository: <https://github.com/taodav/pobax>. An exclusive depth-one
clone is at `tmp/pobax-source-review-01`, ignored through the checkout's local
`.git/info/exclude`. Its tracked files were unchanged after review.

- Commit: [`a5e1d62d14e4efe783885b9d4f19cffa2a568eec`](https://github.com/taodav/pobax/tree/a5e1d62d14e4efe783885b9d4f19cffa2a568eec), dated 7 April 2026, `Merge pull request #38 from taodav/navix_rs_fixes`.
- License: [Apache-2.0](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/LICENSE); file SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
- [`pobax/envs/jax/rocksample.py`](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py): SHA-256 `88a73c93944b0a9008b726b86a8e2260aecc54b7756a7d805019a726676cfd78`.
- [`pobax/envs/__init__.py`](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py): SHA-256 `eae213739f4ad7b7d8e1afd57de4bf334358055fa07ea0ea813080f44a345749`.
- [`pobax/envs/wrappers/gymnax.py`](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/gymnax.py): SHA-256 `64b7d8c0973ad968292016076d79f06dc35dfe418537c76e3117590a5d65ce83`.
- [11-by-11 configuration](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/configs/rocksample_11_11_config.json#L1-L9): SHA-256 `00b3b1fa07a9115885a8def0e7e7e5be3a9c5bcab46f3e49371f33d8914ce85c`.

## Actual observation and action contract

| Component | Pinned implementation |
| --- | --- |
| Standard observation | Length `2*n+k`, hence 33 for `(11,11)`: two one-hot position vectors and `k` rock channels. A check produces exactly one signed reading, `+1` or `-1`; all other rock channels are zero. Move, sample and reset observations have all-zero rock channels. **No previous reading is carried.** [Observation construction](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L197-L250). |
| Actions | `0..3`: north/east/south/west, clipped to the board; `4`: sample here; `5..15`: check rock indices `0..10`. Checking and sampling do not move the agent. [Action space](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L127-L164), [transitions](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L252-L305). |
| Sensor | A reading is correct with probability `(1 + 2**(-distance/20))/2` for the pinned configuration. Distance is Euclidean distance to the indexed rock. The 20 is a configured half-efficiency distance, not the board's maximum distance. [Likelihood](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L19-L21), [check](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L236-L250). |
| Reward and state change | Sampling a good rock yields `+10`, a bad/depleted rock `-10`, and an empty square zero. Sampling sets that rock's quality to bad. Moving onto the eastern boundary terminates and gives `+10`. There is no base movement/check penalty; discount and timeout still matter. [Sampling and exit](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L261-L305). |
| Map and hidden state | Labeled rock coordinates are drawn without replacement at construction and stored in `self.rock_positions`. They are absent from ordinary observations and base `info`. Reset redraws independent Bernoulli quality bits and agent position, keeping the same map. The returned simulator state contains `rock_morality`; it is not an actor input. [Construction/map](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L145-L195), [reset](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L223-L234), [base output](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L307-L318). |

The supplied PPO configuration enables previous-action concatenation, adding
16 one-hot channels. Its actor receives observation and episode-boundary flag,
not previous reward or simulator state. The factory wraps reward normalization
even when general normalization is off; `LogWrapper` retains raw reward and raw
returns in `info`. A future comparison must distinguish actor inputs, training
reward normalization and raw evaluation returns. [PPO actor](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/algos/ppo.py#L55-L70),
[action concatenation](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/gymnax.py#L607-L646),
[factory wrappers](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py#L284-L319),
[raw logging](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/gymnax.py#L154-L180).

## Paper and implementation are not interchangeable

The [paper, Appendix C.3](https://cs.brown.edu/people/gdk/pubs/pobax.pdf)
describes a latest-reading memory reference. At this commit:

- `get_env(..., perfect_memory=True)`, used by PPO, selects
  `RSFullyObservableWrapper`: the observation contains the **true current quality
  of every rock**. This is a privileged-quality reference.
  [Factory selection](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py#L251-L263),
  [true-quality access](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L93-L125).
- `get_transformer_env(..., perfect_memory=True)` selects
  `RSPerfectMemoryWrapper`, which carries the most recent nonzero reading. It
  does not integrate repeated noisy evidence. On sampling it uses the hidden
  map to determine which rock channel becomes `-1`; therefore even this wrapper
  is not wholly reconstructible from the default actor stream without map
  knowledge. [Transformer factory](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/__init__.py#L396-L407),
  [carry and map access](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L62-L91).

The paper also describes binary rock features and a maximum-distance denominator;
the pinned code uses signed readings and configured distance 20. The current
`best` PPO script uses learning rate `0.0025`, lambda `0.7`, width 256, eight
environments and five million steps, whereas the paper's Table 5 RNN settings
list learning rate `0.00025`, lambda `0.95`. This review establishes neither
which revision produced the paper's results nor a reproduction of them.
[Current configuration](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/scripts/hyperparams/rocksample/best/rocksample_11_11_ppo_best.py#L5-L32).

## Can a sufficient exact filter use only public history?

**Not the usual small independent-quality filter under the current default
interface.** Its sensor likelihood needs the indexed rock's coordinates, and
its depletion update needs to know which indexed rock occupies the sampled
square. Neither identity is directly supplied by the ordinary observation.
Reading `env.rock_positions`, a simulator state or the map-generating seed would
add information beyond that stream.

If the map is explicitly given equally to all agents as known task information,
then an exact quality filter is straightforward: initialize eleven probabilities
at `0.5`, perform a Bernoulli likelihood update on a check, set the sampled rock's
probability to zero, and reset at episode boundaries. This is a **different
declared input contract**, not a baseline already qualified here. With unknown
map, exact Bayesian filtering is possible in principle over the joint posterior
of distinct labeled locations and qualities, but eleven independent quality
marginals are not sufficient. An approximate map/belief model is another baseline,
not an exact oracle. Reward-assisted map inference would also add an actor channel
absent from the supplied PPO actor unless provided equally to every method.

Source-based aliasing is real: two histories with different earlier check
results can end with the same movement sequence longer than a chosen recent
window, within the episode horizon, and the same current observation while
implying different quality beliefs. This is a constructive
reason to investigate memory, not evidence of a measured policy-value gap.

The episode contract still needs qualification. The time-limit wrapper resets
and returns a new observation on truncation. Natural termination additionally
depends on the unpinned Gymnax `Environment.step` dependency. The legacy carry
wrapper merges prior memory without checking `done`; if the underlying step
auto-resets, cross-episode carry is a possible defect, not a reproduced result.
Never assimilate every `done` observation as evidence about the terminating
world. [Time-limit lifecycle](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/wrappers/gymnax.py#L211-L239),
[carry update](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py#L58-L91).

## Feasibility and next useful qualification

The environment is vector state, with no renderer required for stepping. A GRU
implementation and CPU entry point exist. JAX officially supports Apple ARM CPU,
but that is not proof this complete dependency graph works locally. The project
requires Python >=3.10; `requirements.txt` pins JAX 0.6.2 but otherwise leaves much
unlocked and points Navix at an unpinned Git branch. Ordinary imports pull in
Brax/Navix even for RockSample. Resolve and freeze an isolated dependency set
before any import check; do not assume the current OpenJev environment is suitable.
[GRU source](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/models/network.py#L10-L37),
[CPU example](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/README.md#L115-L126),
[dependencies](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/requirements.txt#L1-L16),
[JAX installation documentation](https://docs.jax.dev/en/latest/installation.html#supported-platforms).
No local speed, memory, installation or learning estimate was measured.

The next useful artifact is a public-input and episode-boundary qualification,
followed by an opportunity check using the unchanged task: current observation,
recent observation/action windows, explicit public-history memory and a strong
GRU. Give privileged-quality controls their own label. Pin map identities
separately from episode/learner seeds: upstream creates the map before training
and evaluates by resetting that same environment, so default evaluation does not
establish unseen-map transfer. [Environment creation](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/algos/ppo.py#L160-L174),
[same-environment evaluation](https://github.com/taodav/pobax/blob/a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/algos/ppo.py#L423-L442).

**Disqualifier:** if recent windows or a simple matched belief/memory reference
capture the useful gain, do not invent a larger recurrent world model. The task's
motion is deterministic and latent qualities change only through sampling;
active belief maintenance has a clearer opportunity than learning rich physical
dynamics. Avoid repeating OpenJev's cases where literal carry, recent-five support
or linear prediction explained the improvement. [Existing negatives](experiment-index.md).
This is an evidence note, not an execution protocol. It does not reopen the
closed pooling study.
