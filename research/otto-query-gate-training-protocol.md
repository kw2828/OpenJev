# Matched recurrent and stateless gate fitting

Freeze the completed [collection](otto-query-gate-collection-protocol.md), its
successful original supervisor, this protocol, model/training sources,
engineering evidence and runtime before training. Use every real row of all 60
episodes. Do not train on incomplete collection or any EVAL trajectory.

Fit GRU32 plus scalar output (6,273 parameters) and a 31-190-1 Tanh MLP (6,271
parameters), three paired seeds 40101, 40102, 40103. Both consume exactly the
same 31 features. The MLP has no hidden state; GRU state starts at zero per
episode. Initialize scalar output weights to zero and bias to float32(log(19)).
The fixed deployment decision is sigmoid(logit) >= 0.05, including equality.
Do not tune this threshold or choose intermediate checkpoints.

Use CPU float32, one numerical thread, 80 epochs, Adam learning rate 0.0003,
gradient norm clipping at five, and binary cross entropy with logits. For N
rows and episode length T_e, row weight is N/(60*T_e). Report the weighted
sum divided by N, giving each complete episode equal weight. No class or gap
filtering. Fit-order and epoch-order choices are frozen in the plan.

For each fit use one NumPy default_rng(seed + 20000), consuming successive
epoch permutations identically across families. Batch eight episodes and
process chronological 32-step windows, carrying and detaching GRU state between
windows. Padding has zero loss; retain every real tail row. Each optimizer
step divides its weighted real-row sum by the fixed 256 slots. Match all real
row exposure, permutations and update counts across the paired families.

Save each initialization and final model as safe float32 NPZ tensors with
explicit kind, seed, version and threshold metadata. No pickle or implicit
remote code. Save per-epoch metrics, complete optimization counts and costs.
The six final models are the only candidates eligible for evaluation.

Deployment uses a declared NumPy implementation in the original native runtime,
avoiding installation of Torch into that frozen TensorFlow environment. Before
admission, replay every complete TRAIN sequence from every saved final model
through independent evolving Torch and NumPy states. Require maximum absolute
logit/state error <= 2e-5 and identical fixed-threshold query decisions at every
real row. Retain the parity witnesses. This is tolerance-based deployment
qualification, not bit-identical framework arithmetic. Do not widen tolerance
or revise models after observing a parity failure.

One original suspend-inclusive supervisor permits 300 seconds, one CPU thread,
4 GiB RSS and 2 GiB output. Together with collection's 600-second cap this
preserves the prospective 900-second total allocation. Runtime includes loading,
framework setup, all six fits, final deployment replay and evidence. No model
selection, replacement seeds, budget extension or automatic scientific retry.

Completion requires all six fixed fits, matched exposures, finite results,
all deployment checks, unchanged frozen inputs/sources and successful parent
closure. Training fit or deployment parity does not establish task performance.
