# Conditional model design: error-corrective episodic memory

**Research proposal only, now blocked for this recipe.** The completed Mystery Path qualification passed 16/17 checks, with full-public-memory success 195/256 (76.171875%), below the frozen 80% requirement. The recipe is closed. Its memory contrasts do not authorize training, planner repairs, new layouts, or relaxed admission. This note records one transferable mechanism for a separately qualified task; no model, environment, training, or random-stream calls were made in preparing it.

## One hypothesis, with close prior art

Test whether **an error-corrective associative write retains observed negative evidence through repeated visits better than an additive write**, at the same learned projections and matrix-state size. The target is interference in learned memory, not another scalar surprise gate, biological branding, or a replacement for an exact map.

Three primary sources bound the idea:

- [Neural Map, Parisotto and Salakhutdinov, arXiv:1702.08360v1](https://arxiv.org/abs/1702.08360v1) already uses learned spatial memory writes for partially observed navigation. A location-indexed learned map is therefore a required conventional comparator, not our contribution.
- [Linear Transformers Are Secretly Fast Weight Programmers, Schlag et al., arXiv:2102.11174v3](https://arxiv.org/abs/2102.11174v3) explicitly replaces additive fast-weight writes with a delta rule that corrects the stored key-value association. Its retrieval, translation, and language results establish the update's prior art, not sensorimotor control efficacy.
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length, Yang et al., arXiv:2406.06484v6, Section 2.2](https://arxiv.org/html/2406.06484v6#S2.SS2) gives the matrix recurrence and scalable training. Its normalization, state-size comparisons, and stated length-generalization limitations caution against assuming a tiny CPU implementation inherits large-model results or unlimited memory.

**No architectural novelty is established by combining these mechanisms.** The worthwhile question is whether a particular write rule protects actionable evidence under interference, and whether that improvement survives strong ordinary memories and real control costs.

## Minimal candidate and causal contract

Use a 16-by-16 float32 episode-local matrix `S`, initially zero. A slow encoder maps a public event address to an L2-normalized key `k`; a second encoder maps its actually observed binary outcome to a value `v`. A learned scalar write rate is `beta = sigmoid(b(event))`:

```text
read before observation: r = S @ query(public observation, issued-action candidate)
predict:                 p(failure) = sigmoid(decoder(r, public query))
write after real result: S <- S + beta * outer(v - S @ k, k)
```

Here the event address must be recoverable from public observations and issued commands, such as the observed destination of an attempted move. It cannot contain a hidden route, seed, simulator coordinate, future outcome, or evaluator progress index. The key used to query a destination and to write its returned evidence must have the same addressing convention. There is no one-hot safe/unsafe table supplied to the candidate. Repeated public observations may write repeatedly; no privileged first-visit flag suppresses them.

Predict an actual transition before assimilating its returned outcome. Write the resulting public evidence once, including negative evidence, after the real transition. A within-episode return to a starting location preserves `S`; an actual episode reset clears it. A forced return must not be mistaken for the submitted movement. Candidate queries are read-only; any future imagined branch receives a private copy and cannot train or overwrite deployed memory. The slow parameters are fixed at deployment: these fast writes are recurrent inference, not deployment optimizer steps.

The future task must make a past public fact meaningful when queried again. This is not a valid stale-label objective if that fact can change unobservably. Unknown locations never receive hidden labels. A deterministic motion skeleton may be supplied equally to every model to isolate memory, but then the experiment tests a learned belief about outcomes, not discovery of physical dynamics.

## Bounded development pilot, only after separate task admission

The following is a proposed fixed screen, not an executable protocol for closed Mystery Path:

- Collect public trajectories on **128 training layouts and 32 development layouts**, disjoint by whole layout and from any task-qualification cases. Use two fixed public behavior controllers, full explicit memory and a 32-transition controller, in four fixed tie orders: 1,024 training and 256 development episodes. Retain successes and failures. Freeze collection limits and public event semantics for the independently qualified task; do not transplant an unverified horizon.
- Train **eight arms, three paired initializations each, four epochs, batch 32**, hence **128 optimizer updates per fit and 3,072 total updates**. Use complete padded episodes with valid-prefix masks, float32, Adam learning rate `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, gradient-norm clip 1. No early stopping, architecture sweep, or development-based checkpoint selection. All 24 fits finish before development metrics are read. A measured capacity check must set wall/storage caps before any collection or fit.
- The arms are delta writes; additive outer-product writes; delta with failed-event writes disabled; delta erased on within-episode restart; dense GRU64; dense GRU256; a GRU64 rebuilt from at most 32 actual transitions; and a conventional learned spatial memory with four features per publicly addressable location. Delta/additive/erasure/write-removal variants share exact initial tensors and compute the same projections. The other architectures use paired data and orders, with actual parameter/state sizes reported rather than falsely declared identical. GRU256 matches the matrix's 256 floating state values, but not its parameter count or operations.
- Use **unweighted Bernoulli negative log likelihood on actually observed next failure cues**, averaged over valid transitions per episode and then episodes. Stop gradients through measured targets, retain gradients through the episode's writes, and mask padding without inventing observations. Do not add reward bonuses, teacher hidden maps, uncertainty weighting, or a special training reward for the proposed mechanism.

The primary development diagnostic queries previously observed public facts before the current write. Include all eligible negative facts whose last observation was at least 32 real transitions earlier; facts must remain semantically valid in the chosen task. Average Brier error over facts, then query boundaries, then episodes and layouts; report all denominators and all positive-fact errors separately. If fewer than 16 development layouts contain eligible negative facts, the diagnostic is underpowered for this hypothesis: stop rather than manufacture longer delays after seeing results. The full explicit map is a deterministic reference using the same public evidence, not a learned arm.

Proposed admission to a later controller test requires the delta family to reduce this Brier error by at least **0.01 absolute versus additive writes**, improve by at least **5% relative versus each ordinary learned comparator**, and be nonworse in every paired fit. A comparator at zero error blocks a claim of relative improvement. The erasure and failed-write-removal ablations must each worsen the negative-fact metric by at least 0.01; otherwise the proposed explanation is unsupported. Overall executed-transition NLL may regress by at most 2% versus each comparator. Freeze these comparisons before training; retain every failed condition and fit. This is a development screen, not test-set efficacy or calibrated confidence.

## What would justify further research

Record actual parameter counts; matrix, recurrent, spatial and raw-history bytes; reconstruction and query counts; all writes, forward/backward/optimizer calls; padding work; data collection and parsing; scratch memory; serialization and full training/inference wall time. The 256-float matrix alone uses 1,024 bytes. An exact small symbolic map can require far less: lower Python object overhead is not an information-compression or architectural advantage. Likewise, better accuracy from a larger stored state is not a write-rule result.

Only a successful prediction screen could justify a separately frozen control comparison in that independently admitted task. Before adding a second task, require a practical held-out control benefit over the strongest ordinary learned memory, preserve all paired fits, and retain the public-map reference and whole-cost comparison. A concrete proposed threshold is at least five percentage points in native success with no paired-fit regression, while remaining within five points of the explicit public-memory controller. This threshold would be frozen before that control test, not chosen from its results.

If additive writes, ordinary spatial memory, or finite history explain the benefit, report that and stop the mechanism claim. A second task would need different observation/action semantics and a predeclared transfer or interference question, rather than another size setting selected to favor the matrix. There is presently no basis for a fly-connectome claim or an ICLR novelty assertion, and no permission to reopen the failed qualification.
