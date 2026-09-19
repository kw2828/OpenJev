# POPGym Concentration: direct learned-memory pilot

Source review completed 2026-09-19 UTC. This is a prospective design, not a benchmark result. No POPGym package was installed or executed by this reviewer; no episodes, models, random seeds or fitted checkpoints were used.

**Recommendation:** use native `popgym-ConcentrationHard-v0` for a direct supervised public-memory pilot followed by native closed-loop evaluation. Train the proposed models, retain the strong GRU and exact public-map controls, and report all results. Another rule-based competence threshold is not a prerequisite. This does not reopen the stopped MysteryPath or robotics qualifications.

## Pinned native contract

The reviewed official repository revision is [`410d5aa626dae8024f498354d8781a0d1870c399`](https://github.com/proroklab/popgym/commit/410d5aa626dae8024f498354d8781a0d1870c399), dated 2026-06-11. The controlling files are [Concentration](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/envs/concentration.py) and [Deck](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/core/deck.py).

| Native difficulty | Cards | Categories | Hidden token | Maximum actions |
| --- | ---: | ---: | ---: | ---: |
| Easy | 52 | 2 colors | 2 | 104 |
| Medium | 104 | 2 colors | 2 | 208 |
| Hard | 52 | 13 ranks | 13 | 104 |

Hard has four cards of each rank, not 26 unique pairs. The observation is a length-52 integer array, each entry a rank 0 through 12 or hidden token 13. The action is a card position 0 through 51. Reset uses the environment's seeded shuffle, returns all hidden cards, and empty `info`. No seed is a model input.

The returned observation is captured **before** resolving the current pair. A failed second flip therefore returns both revealed ranks, although the next action starts a new pair. A successful pair stays visible permanently. Do not obtain a second observation after `step` to remove a mismatch: that changes the native public interface. On every action, `after[action]` reveals its rank, including a mismatch or repeated selection.

An ordinary first flip earns 0. Matching two distinct cards earns `2/52`; an unsuccessful second flip earns `-2/104`. Selecting a permanently matched card immediately clears the pending pair and costs `-len(in_play)/104`, where `len(in_play)` is one or two. Selecting the pending position again is not a match. The final budgeted action still executes and earns its reward. Stop on `terminated or truncated`; both can be true together. A perfect board-aware agent can earn 1 in 52 actions, but that is not the performance of a public-memory agent discovering an unknown board. Native return ranges from -1 to 1. These details follow the pinned step implementation, not an alternative concentration game.

The entire board is exposed by `get_state()`, `.state`, deck internals and several [Markovian wrapper modes](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/wrappers/markovian.py). They must never enter collection-policy, model or teacher inputs. An audit may access privileged state in a separate namespace, but the proposed supervised labels need none of it. Public action/reset indicators are legitimate; the official benchmark uses [PreviousAction](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/wrappers/previous_action.py) and [Antialias](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/wrappers/antialias.py), the latter distinguishing the reset placeholder from a real action 0.

## Collector, labels and shared decision rule

Use whole native Hard episodes, up to all 104 actions unless the board is completed sooner. Save reset observation and every actual action, reward, returned observation, termination and truncation flag. Retain failed prefixes separately. Collect a fixed mixture of 50% exact public-map decisions and 50% uniform legal decisions, with an explicitly supplied behavior generator. This is a proposed behavior distribution, to freeze before any model results. It is not a privileged-deck teacher.

Keep two separate objects:

* **Public tracker:** current public frame, seen-position flags, permanently matched flags, pending position and action count. It must not retain historical rank values. Each actual action produces exactly one reveal `(position, rank)` for the model. Queries do not write memory.
* **Training-only public teacher:** a position-to-rank table populated only from earlier returned public frames. A target is valid only when that position has been seen and is hidden in the current frame. Currently visible and never-seen entries are excluded from the primary recall loss. The table is available to the loss/evaluator and explicit-map baseline, not as a model feature or shared learned-policy cache.

All models use the same picker. Replace predicted distributions for currently visible positions with public one-hot ranks, and for never-seen positions with the same uniform 1/13 distribution. For a pending first flip, select the unmatched, different position with greatest probability of that visible rank. Otherwise select the first position of the unordered unmatched pair maximizing the dot product of their rank distributions. Break ties by lowest position, then lowest partner. Recompute the second action after observing the actual first reveal. Exclude already matched positions and the pending position; use the same exclusions and metadata for every model and the explicit-map reference. The native environment itself still permits these wasteful actions, so this is a declared shared controller constraint, not a native action-space change.

This setup tests learned storage and retrieval of prior public associations, and whether retrieval supports native decisions. It does not test unaided end-to-end policy learning. The visible-frame override and shared seen/matched flags are meaningful external memory aids; report their bytes, time and semantics alongside each learned state.

## Bounded genuine learning comparison

The parent's bounded initial scope is 128 collection episodes for training and 32 disjoint development episodes, all native Hard. This is a development pilot, not an untouched scientific test. Freeze whole-episode membership and ordering, behavior, model families, optimization budget, final-checkpoint selection and evaluator before fitting. A small engineering capacity check can set a wall cap without examining model quality. Fit all models before opening final development outcomes; do not substitute an easier difficulty after failure. A later scientific evaluation would need a separately frozen unseen cohort.

Use three initialization pairs for the six implemented families: **GRU**, **delta**, **gated delta**, **diagonal KDN**, **innovation-local**, and **innovation-matched**. The matched control spreads inflation uniformly but matches the local rule's gain along the current key from the same input state and event. It does not match added trace, resulting diagonal state or later trajectories. Covariance reset is not an implemented arm. These are 18 fits, without a parameter-count or compute-matching claim. The exact public-map reference needs no learned fit; the bounded last-32-selected-reveal reference checks a simple limited-history alternative, with the same current-public-frame override.

The main supervised objective is cross-entropy on previously observed, currently hidden ranks, using the same labels, examples and order for all fits. Report hidden-seen recall accuracy and cross-entropy, including lag strata fixed before results; report initial and final checkpoints. Closed-loop primary outcomes should include paired native return, matched pairs and completion rate across every fit, with all cases retained. Also report repeated/invalid choices if any, action counts, per-fit variation, state sizes and whole-episode time. Conditional case uncertainty is not uncertainty over arbitrary training seeds. Architecture matching by a shared encoder/readout or state size does not imply matched parameter count or wall time.

Avoid a new arbitrary expert-success admission gate. Freeze comparisons and practical effect thresholds prospectively if a later continuation decision needs them, then report success or failure honestly. A first learning pilot can establish trainability and expose a failure mechanism without being a novel architecture result, an ICLR claim or a new POPGym state of the art.

## Relation to the original benchmark

The [ICLR 2023 POPGym paper](https://arxiv.org/abs/2303.01859) evaluated recurrent PPO with three trials, usually 15 million environment steps per run, and 256 recurrent-state scalars. Its reported MMER selects the maximum mean episode return over training epochs. ConcentrationHard was difficult: GRU MMER was about -0.829 and MLP about -0.833. The paper's historical code revision is `e397e5e`, whereas this review pins the current Gymnasium implementation. Our public supervised-memory/common-picker pilot is a different training and evaluation regime, so those figures are context, not a direct performance comparison.

The current [POPGym fast-weight implementation](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/baselines/models/fwp.py) explicitly omits the delta update and DPFP. It is not a Gated DeltaNet baseline. [Gated Delta Networks](https://arxiv.org/abs/2412.06464) is an established comparator, not a new mechanism introduced by this pilot. The legacy RLlib launcher and synthetic throughput benchmark are not evidence of OpenJev runtime, matched compute or reproducibility under the current Python environment.

## Alternatives, without switching to chase a pass

[RepeatPrevious](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/envs/repeat_previous.py) asks for a past suit with fixed delays 4/32/64. At decision time with observation `o_t` already available, the scored target is `o_(t-k+1)` once available. The first `k-1` actions are unscored; actions do not affect the observation stream. It is a useful FIFO/delay diagnostic, but lacks Concentration's action-dependent discovery.

[Autoencode](https://github.com/proroklab/popgym/blob/410d5aa626dae8024f498354d8781a0d1870c399/popgym/envs/autoencode.py) presents 52/104/156 suits and then requires reverse-order recall. The final revealed card arrives with the PLAY flag and is the first answer, so that first scored action is not long-term recall. WATCH actions do not influence data; an exact public stack is the reference. It can isolate sequence storage, but should not replace Concentration merely because it is easier to supervise.

## Minimal implementation boundary and source identity

Add an environment-free public tracker/teacher and picker, a native collector with explicit generators, a shared fitted-memory interface, and a closed-loop evaluator with source-bound receipts. Preserve native returned arrays and action timing. Do not reuse robotics state validators or import the old RLlib training stack. Independent saved-output audit can check public label causality, membership, action legality, return arithmetic and native replay; it cannot prove neural gradients from output files alone.

Verified raw-source SHA-256 values:

| Pinned file | SHA-256 |
| --- | --- |
| `popgym/envs/concentration.py` | `23c010fe308ce960ac08af49666423ecd200449d8743c101f4d80ff223c43f99` |
| `popgym/core/deck.py` | `bca5004bdec427461226db6484369793b2e71a17ce01e32fa83a1f8d42d2934d` |
| `popgym/envs/repeat_previous.py` | `a3c135de949adac62388efa00cc9bed2e047c4f431f3dd6d2cdfca3247c27b3c` |
| `popgym/envs/autoencode.py` | `e8b3d17b3d32fb8565eedccdb70eb0e46789a449343b03138339b7ffd47612db` |

The source review establishes interface semantics and a feasible experiment design. It supplies no learned-memory or native-control result.
