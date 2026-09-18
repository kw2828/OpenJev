# Connectome chess study

This study asks whether biological wiring helps a trained chess policy after controlling for its representation, recurrent computation, signed degrees, input mapping and training budget. It follows the failed candidate-conditioned study. No biological-topology performance advantage has been demonstrated.

The [frozen protocol](../evidence/chess-connectome-v1/protocol/plan.json) completed and its original report audit passed on September 18, 2026. The biological topology failed the continuation rule: only 5 of 30 checks passed. Biological mean bounded loss was 0.10198/0.10396 (ordinary/shifted), compared with 0.09099-0.09664/0.09741-0.10426 across the three rewires. No topology advantage is established. [Results](../docs/chess-connectome.md).

Its exclusion snapshot contains 3,826,350 prior natural board states, with mirrored states reserved during admission. [Protocol receipt](../evidence/chess-connectome-v1/protocol/prepared.json).

The [adapter](chess-connectome-adapter-pilot.md) uses all 1,409 descending neurons and 44,090 signed directed connections from the audited source. This induced graph omits 430,912 boundary connections and sensory inputs. Its board interface is artificial; it is not an intact fly brain.

## Frozen comparison

For each of the three published direct-policy seeds, freeze the entire chess backbone and fit six adapters: biological wiring, three independent signed-degree-preserving rewires, dense recurrence and node-local recurrence. Evaluate those 18 final adapters alongside all three unchanged backbones. Biological and rewired arms have matched parameters and operations. Dense and node-local arms are additional quality/cost references with different parameter counts.

Every fit uses the original 32,768 training positions, six epochs, batch 128 and exactly 1,536 Adam updates. Within a seed, the minibatch order and board-to-node mapping are identical across arms. Every fit finishes before any neural evaluation starts. There is no best-epoch selection, retry, replacement seed or budget extension. Backbone pretraining remains part of the recorded cost.

Two fresh development panels each contain 2,048 positions. Admission excludes historical roots, all legal successors and mirrors, including the **entire 62,561-row ChessBench source file**, not just its selected transfer panel. New roots and their legal successors are also disjoint across the two panels. The generators remain previously studied development generators.

Record every legal move's raw score and check target agreement, negative log-likelihood and value error. Grade fixed subsets of 128 positions per panel with Stockfish 19 at 20,000 nodes per call. Engine loss is the unrestricted score minus the chosen-move score, bounded by `tanh(cp / 600)`. Finite-search negative differences are retained. This is an engine-relative measurement, not game-theoretic regret.

Measure full CPU decision latency on a fixed 64 positions per panel, with three separately charged starting-position warmups for each model. Record host load before and after timing on this shared desktop; these are not isolated hardware measurements. All topologies currently use dense matrix operations; sparse wiring is not evidence of sparse-kernel speed.

## Continuation rule

Advance only if all of these hold:

- Biological wiring reduces mean bounded engine-score loss by at least 10% against **each** of the three rewires on **both** panels, with positive comparator means.
- Biological wiring has lower loss in **every paired seed** against each rewire on both panels.
- No paired seed degrades against its unchanged backbone on either panel.

The executable rule uses a fixed arithmetic tolerance of `1e-12`. Missing, failed or nonfinite evidence cannot pass. Means and paired differences are descriptive; they are not a population-level significance test.

Stage 1 contains no arena or Elo estimate. A pass would justify a separately frozen arena replication and a targeted intervention on the responsible wiring property. It would not establish novelty, world-model learning or game-playing superiority. The graph state resets each decision; this is recurrent computation without cross-move memory or learned transitions.

## Reproduce

The executable entry point is [chess_connectome_study.py](../scripts/chess_connectome_study.py). It binds the source code, tests, original checkpoints, graph identities, engine, environment, data exclusions and protocol before training. `report` audits saved outputs without new model or engine calls.

```sh
.venv/bin/python scripts/chess_connectome_study.py prepare --out evidence/chess-connectome-v1/protocol
.venv/bin/python scripts/chess_connectome_study.py verify --plan evidence/chess-connectome-v1/protocol/plan.json
.venv/bin/python scripts/chess_connectome_study.py run --plan evidence/chess-connectome-v1/protocol/plan.json --out runs/chess-connectome-v1/execution
.venv/bin/python scripts/chess_connectome_study.py report --plan evidence/chess-connectome-v1/protocol/plan.json --execution runs/chess-connectome-v1/execution --out runs/chess-connectome-v1/report
```

Preparation needs the pinned local inputs and historical receipts; cloning the source alone is insufficient. Output paths are exclusive. Graph arrays and derivative checkpoints remain local under `runs/`, separate from the MIT code release and subject to the source data's terms.
