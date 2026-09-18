# Connectome residual-adapter pilot

Status: the [architecture prototype](../src/openjev/research/chess_connectome_adapter.py) and [controlled training study](chess-connectome-study.md) completed. The original report audit passed on September 18, 2026; the biological topology failed its frozen continuation criterion (5/30 constituent checks). [Results](../docs/chess-connectome.md). The earlier full-size CPU/MPS preflight completed six cases and 18 synthetic updates. This study remains separate from the failed candidate-conditioned comparison and the completed ChessBench transfer panel. No biological-topology advantage is established.

## Question

Does the higher-order wiring of a small biological graph improve a chess policy after controlling for the board representation, node identities, signed degrees, input/readout mapping, training objective and update budget?

Use the [audited descending-neuron graph](../evidence/connectome-graph-v1/README.md): all 1,409 nodes, 44,090 signed directed edges and six isolated nodes. Three independently rewired controls retain the same nodes, per-node signed degrees and coarse-group composition. No chess score selects the graph or its rewires. Induction removed 430,912 boundary edges and all sensory-input nodes. This is a wiring-prior experiment with an artificial interface, not an intact functional fly circuit.

## Shared chess interface

Use every published direct checkpoint, seeds 97, 109 and 127. Freeze the complete width-32 backbone, including its four root computation steps and policy/value readouts. No best checkpoint is selected. Retain its pretrained feature map as the residual path.

Project each square's 32 features to 16 channels. A seeded, balanced assignment maps original graph-node identities to square/channel slots; keep the assignment identical across topology arms within a paired seed. Initialize scalar node states as `tanh(drive)` and perform four leaky, signed graph updates with a fixed mixing factor of 0.5. Each proposed update is `tanh(drive + normalized_messages + node_bias)`. Learn fresh positive edge magnitudes via softplus, initialized to 1.0, normalized by fixed incoming degree. This starts each nonempty row with absolute incoming edge mass one; the tanh/leaky updates bound node states. Node biases start at zero. Do not use anatomical counts as initial weights.

Before any full-size preflight or training, the initial edge magnitude was changed from the prototype's 0.1 to 1.0 to avoid an additional tenfold suppression after degree normalization. This was an analytical design change without chess-performance feedback; the zero output projection still preserves exact baseline predictions at initialization.

Pool node-state changes back into the same slots, averaging by occupancy. Project them to 32 channels and add them to the original feature map before the frozen readouts. Initialize the output projection to zero so every arm begins with exactly the same direct-policy output. Report the initial delayed gradient into the graph while this projection is zero; verify that the graph receives gradients after the output projection changes.

Each decision starts with fresh graph states. These updates do not predict a next board, carry memory between moves or implement search. They are not a recurrent world model.

## Comparisons and proposed budget

For each of the three paired backbone/mapping/training seeds, fit six adapters:

- Biological topology.
- Three independent signed-degree-preserving rewires, using the audited fixed seeds 151, 163 and 179.
- Dense recurrence over all directed pairs, including self edges, with fixed seeded random signs as a quality-cost reference.
- Node-local recurrence with a positive learned self edge per node and no communication between nodes.

Also evaluate the unchanged frozen backbone. The three rewires are the primary topology controls. Dense and node-local recurrence have different active parameter counts; they cannot isolate topology by themselves. A no-adapter comparison alone also changes depth, parameters and computation.

The proposed training budget is the original 32,768 training positions, six epochs, batch 128 and 1,536 updates per fit, with policy cross-entropy plus 0.5 times value MSE. This yields 18 final fits. Freeze the optimizer, minibatch orders, all data and checkpoint hashes, graph construction and mappings before training. Count the original backbone pretraining in the evidence record. Reusing a fully pretrained backbone does not establish low-data sample efficiency.

Use fresh ordinary and shifted development panels with exclusions covering all historical roots, legal successors and mirrors, including the ChessBench transfer exposure. Report every seed, target-move agreement, bounded engine-score loss and full decision cost. The executable protocol must define the fixed engine-grading subset and any paired arena before fitting; this document is not that executable freeze.

## Proposed continuation rule

Advance only if the biological adapter reduces mean bounded engine-score loss by at least 10% against **each** rewired control on **both** panels, with positive comparator means. Require a lower loss in every paired seed on both panels and no degradation against the unchanged direct backbone. Otherwise report no demonstrated biological-topology advantage. Freeze the precise aggregation and failure handling before training; do not revise the rule from observed scores.

A successful pilot would justify a larger replication and an intervention on the wiring property responsible. It would not alone establish architectural novelty, an Elo rating, sample efficiency or a world-model benefit. [Prior work and scientific limits](chess-connectome-followup.md).

## Implementation checks

Synthetic tests verify exact initial outputs against the full direct backbone, independent backbone storage, frozen weights across optimizer steps, immediate output-projection gradients and later graph gradients. They independently reconstruct signed message direction, degree normalization, recurrence and occupancy pooling, including isolated, parallel and self-edge fixtures. They also check deterministic shared mappings, restored global RNG state, reset between calls, reordered and padded candidate menus, native board/history preservation, CPU float32/float64 behavior, invalid inputs and parameter/MAC counts against actual module calls. The subsequent full-size preflight passed the three modes on CPU and MPS, including exact initialization and frozen-backbone checks. No CPU fallback or retries occurred. These checks do not measure chess quality.

## Runtime and asset handling

The initial implementation materializes dense matrices for every topology. A 1,409-by-1,409 FP32 matrix occupies 7,941,124 bytes. Report actual matrix operations, frozen-backbone work, slot projections, legal-move preparation, training time and full inference time. A sparse parameter mask is not a sparse-kernel speed result. The completed bounded preflight establishes local feasibility before a training budget is frozen; its timings are not isolated benchmarks.

The [official FlyWire guidelines](https://flywire.ai/guidelines) identify public v783 data as CC BY-NC 4.0. Keep raw graphs, induced graphs, rewires and resulting derivative model weights outside the MIT release; exact source notices and a separately attributed derivative release still need to be resolved. Independent implementation, synthetic tests, aggregate measurements and graph hashes can be retained separately. No graph artifacts or trained derivatives are distributed by this prototype.
