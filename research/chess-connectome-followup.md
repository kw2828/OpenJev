# Biological topology as a chess-model prior

Status: graph-only loading, exact induced extraction and matched rewiring controls implemented; 40 synthetic graph tests pass. The [real-asset audit and three-control structural preflight](../evidence/connectome-graph-v1/README.md) completed independently of pretrained weights. The [residual adapter](chess-connectome-adapter-pilot.md) passed 51 synthetic unit tests and six full-size CPU/MPS cases with 18 synthetic optimizer updates. No biological-topology adapter has been trained on chess labels or evaluated for chess effectiveness. This follow-up does not alter the completed candidate-v2 experiment or its success criteria.

## Why this needs a new comparison

Our [original pilot](../docs/chess-student.md) used a synthetic 64-unit circuit. Mean teacher agreement was 14.3555% versus 14.1927% for one signed degree-preserving rewire. The 0.16276-point difference missed the frozen 3-point threshold and changed direction across weight seeds. That pilot did not establish a topology advantage. Its post-hoc feature intervention also showed weak dependence on the supplied board features, motivating a stronger shared encoder before further topology claims.

The published ChessFly adapter uses a real FlyWire-derived graph, but its training, features and compute differ from our students. Its [exhibition results](../docs/chess.md) cannot identify an effect of biological wiring.

## Available graph and reproducible identity

The cached ChessFly Space revision is `375374bf58f1c828ad7d33905b64f1fa6ac4906f`. The packed graph contains 138,639 nodes and 15,091,983 signed directed connections, with 54,492,922 anatomical synapses represented by their absolute counts. There are 9,059,302 positive and 6,032,681 negative connections. These are counts for this exact packed asset, not interchangeable with other FlyWire releases or thresholded graphs.

| Asset | Raw bytes | SHA-256 of raw bytes |
|---|---:|---|
| `connectome.bin.gz` | 91,106,470 | `700a443c885ef2eec25f97136ea14b9a2575fb270321190f4edc4e6f5c49e13d` |
| `neurons.bin.gz` | 1,802,307 | `a245f2f2195ff82a4bd1af8dab70e9d213eca29fd5c7add607b3d983ed7bedfa` |

The neuron asset provides spatial coordinates and four coarse groups: 40,283 other, 86,092 optic, 10,855 input and 1,409 descending. Fourteen coordinate triples are nonfinite. The packed files do not provide detailed cell types, neurotransmitter confidence or an original FlyWire root-ID mapping. Original packed node indices therefore remain the identity available to this adapter. They must not be described as FlyWire root IDs or interpreted as identified functional cell types.

Local provenance is retained in `runs/chessfly-source-audit/meta.json`, `audit-receipt.json` and `assets/download-receipt.json`. Loading graph bytes requires no pretrained tensor weights or model inference. A full dense FP32 adjacency would require 76,883,089,284 bytes before activations, gradients or optimizer state, which exceeds this machine's memory. A small induced-subgraph pilot permits a practical dense control.

The structural preflight used all 1,409 descending neurons, containing 44,090 connections and six isolated nodes after induction. Three fixed rewiring seeds each completed 440,900 accepted swaps and changed 90.83-91.19% of signed edges while retaining signed degrees. Its dense FP32 adjacency would occupy 7,941,124 bytes. Induction removed 430,912 connections to other groups, including all sensory inputs, so this is a tractable structural test rather than an intact functional circuit or a selected training architecture. No chess performance was measured.

## Proposed comparison, not yet frozen

The current residual-adapter proposal uses all 1,409 descending nodes by a fixed coarse-group rule. Record selected node identities, signs, groups, input/readout mapping and the boundary edges lost by induction. Keep the same selected nodes and mapping in every arm. No chess score selected this graph, and the executable training protocol is not yet frozen.

Compare biological wiring with several independently rewired graphs, a random sparse control and dense recurrence. For the primary rewiring control, swap edges only within the same source-group, destination-group and sign bucket. This preserves each node's signed degrees and its incident group composition, as well as total group mixing. The operation changes higher-order wiring while retaining these local statistics. Use topology signs with freshly initialized learned magnitudes; anatomical connection counts are provenance rather than an implicitly preserved strength distribution.

Hold board features, encoder, recurrent dynamics, readout, teacher labels, optimizer, update count and weight seeds fixed. Include a control with the recurrent path removed to test whether the shared encoder and readout explain the result. The sparse arms can match edge parameter count; the dense arm cannot simultaneously match edge count and connectivity, so report its actual parameters and compute. Record accepted swaps and final edge overlap for every control. A swap count alone does not demonstrate sufficient randomization, and repeated swaps need not sample uniformly from all admissible graphs.

Measure teacher agreement and decision quality across training-set sizes, fresh ordinary and shifted positions, paired games and complete inference time. Include graph construction and legal-move handling in runtime. Test multiple topology seeds as well as weight seeds. Keep the already proposed external ChessBench panel separate from the candidate-v2 continuation criteria.

For an action-conditioned world-model comparison, add the same future-prediction target and rollout procedure to every topology arm. Compare policy-only and predictive training within each topology. Hidden-state refinement on a single board does not establish learned future dynamics or memory across moves.

## Prior work and the contribution test

[ChessFly](https://huggingface.co/mlabonne/chessfly) already trains a FlyWire-derived chess policy. [FlyGM](https://arxiv.org/abs/2602.17997) uses the adult fly connectome as a reinforcement-learning controller for simulated locomotion and reports sample-efficiency gains against graph and non-graph baselines. [AlphaGateau](https://arxiv.org/abs/2410.23753) studies graph representations for chess, and [GGNN](https://arxiv.org/abs/1511.05493) establishes gated recurrent graph propagation. Biological wiring, graph recurrence, or chess deployment alone is therefore insufficient as a novelty claim.

A useful result would isolate a reproducible topology-dependent learning or transfer advantage, then identify which wiring properties account for it. If matched rewiring preserves performance, report that outcome. If a dense model has a better measured quality-cost tradeoff, do not describe sparsity as an established efficiency benefit.

## Asset handling

The pinned ChessFly model card declares `license: other` and says FlyWire noncommercial terms apply to graph derivatives; the Space declares GPL-3.0 for its code. The [official FlyWire guidelines](https://flywire.ai/guidelines), checked September 17, 2026, identify the public v783 data as CC BY-NC 4.0. The [license summary](https://creativecommons.org/licenses/by-nc/4.0/) permits sharing and adaptation subject to attribution and noncommercial use. The packed asset separately identifies source-file hashes, including a `LICENSE` file whose contents have not been matched here. Keep cached graph bytes, induced graphs, rewires and resulting weights outside OpenJev's MIT distribution while their exact source notices and derivative-release terms are resolved. Independent graph-control code and synthetic tests can be published without packaging those external assets. This note makes no broader licensing determination.

The [residual-adapter pilot](chess-connectome-adapter-pilot.md) makes the proposed artificial chess interface and matched topology comparison concrete. Its training protocol is not frozen and it has no efficacy result.
