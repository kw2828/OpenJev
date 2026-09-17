# Connectome adapter resource preflight

All six fixed cases completed: sparse, dense and node-local adapters on CPU with batch 8 and MPS with batch 128. Each used a fresh random chess backbone, Gaussian synthetic observations, shaped candidate features and arbitrary targets. Three optimizer updates per case give 18 total. No pretrained checkpoint, real chess position or benchmark label was used.

[Protocol and source hashes](protocol.json) · [Summary](summary.json) · [Byte manifest](manifest.json) · [Script](../../scripts/preflight_chess_connectome_adapter.py) · [Proposed training pilot](../../research/chess-connectome-adapter-pilot.md)

Every case passed exact initial output equality against its direct backbone, finite loss/gradients, learning signals through the residual and graph paths, and unchanged frozen-backbone hashes. The actual 1,409-node descending graph was used only as topology, with fresh edge magnitudes initialized to 1.0. Sparse, dense and node-local adapters have 46,571, 1,987,762 and 3,890 trainable parameters respectively. All three currently execute dense matrix multiplication.

The complete preflight took 15.15 observed seconds. The largest recorded process peak RSS was 1,500,184,576 bytes. These observations include loading, initialization, hash checks and synthetic optimization; other host work could overlap. They establish local feasibility, not a clean inference-speed comparison or chess efficacy. MPS fallback was unset and no fallback or retry occurred. A longer training run can have different memory and runtime requirements.

Files listed in the manifest are byte-identical copies of the local run, including all six case receipts and progress logs. Source files are bound in the protocol; source or protocol changes require a new disclosed run. No graph arrays, node mapping arrays or trained weights are distributed. [FlyWire public-data attribution and terms](https://flywire.ai/guidelines) remain separate from OpenJev's MIT implementation.
