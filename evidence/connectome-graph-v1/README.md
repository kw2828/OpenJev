# Connectome graph feasibility

**Graph controls work; no new connectome chess model has been trained or evaluated.**

The [final graph-only audit](audit-final.json) verifies the pinned external metadata and compressed/raw graph hashes. The packed graph contains 138,639 nodes, 15,091,983 signed directed connections, no self-connections and no duplicate directed pairs. No pretrained weights are loaded.

The [structural preflight](descending-controls.json) takes exactly all nodes in the packed `descending` group: 1,409 nodes and 44,090 induced connections. It tests all three declared seeds without selecting a winner or replacing a control.

| Seed | Accepted swaps | Attempted swaps | Signed edges changed |
|---|---:|---:|---:|
| 151 | 440,900 | 535,231 | 91.1885% |
| 163 | 440,900 | 535,355 | 90.8347% |
| 179 | 440,900 | 534,982 | 90.9118% |

Each control preserves every node's positive/negative in/out degree, node identity and coarse group identity. The implementation additionally constrains swaps within source-group, destination-group and sign buckets; mixed-group synthetic tests verify this invariant. Anatomical strength distributions are not represented as preserved. Every returned graph has the requested completed swap count, and insufficient-budget attempts raise an explicit failure.

This group-induced graph loses 430,912 connections to other groups and contains six isolated nodes. It is not an intact biological functional circuit and has no packed sensory-input nodes. Neither these structural statistics nor a high changed-edge fraction establishes chess quality or uniform sampling over possible graphs.

Code: [graph primitives](../../src/openjev/research/connectome_graph.py), [40 synthetic tests](../../tests/test_connectome_graph.py), [asset audit](../../scripts/audit_connectome_graph.py), [control preflight](../../scripts/preflight_connectome_controls.py). The receipts bind exact source hashes and retain all three control outcomes. Graph bytes, coordinates, selected-node lists, pretrained weights and rewired adjacency arrays are not redistributed here. External asset terms remain separate from OpenJev's MIT code.

The earlier [audit](audit.json) predates a lint-only change to the synthetic tests. The final audit was rerun to bind the current source hashes; both preserve the same graph statistics. No training or evaluation was repeated.

[Proposed scientific comparison and prior work](../../research/chess-connectome-followup.md).
