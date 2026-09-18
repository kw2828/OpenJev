# Is the fixed spatial assignment limiting the connectome adapter?

Status: an implemented isolated prototype and design review, not a frozen training study.
The [original biological comparison](../docs/chess-connectome.md) remains
negative. No current evidence identifies the board-to-neuron assignment as
the cause of that result.

## What already learns

The original interface is only partly fixed. Its shared input projection has
528 parameters, the output projection has 544, and all 1,409 node biases and
44,090 biological edge magnitudes train. The complete root backbone and its
readouts remain frozen. The node-to-square/channel assignment and its tied
occupancy pooling are fixed.

The assignment covers all 1,024 square/channel slots: 385 have two nodes and
639 have one. All 1,409 descending neurons receive drive and contribute to
pooling. The induced graph contains six isolates and excludes 430,912 boundary
connections. There is no preserved sensory pathway that a learned assignment
could simply restore.

A [saved-parameter audit](../evidence/chess-connectome-parameter-audit-v1/summary.json)
checks all 18 original fits under their original checkpoint and execution
hashes, without model or engine calls. Every output projection is nonzero.
For the three biological fits, mean input-weight movement RMS is 0.10591,
output-weight RMS is 0.13020 and edge-gain movement RMS is 0.08746 from the
initial gain of one. The mean fraction of edge gains changing by more than
1e-6 is 99.9992%. This rules out a literally unchanged adapter. Parameter
movement does not establish causal decision use, successful optimization or a
mapping bottleneck; norms across differently sized controls do not measure
functional importance.

## A narrower intervention

Keep each node's channel and learn one hard permutation of the 64 board
squares, shared across all 16 channels. Use the same assignment for gathering
drive and scattering the final-minus-initial graph state. This preserves hard
local sampling and occupancy counts and creates no dense input bypass.

Let S be the node-to-slot assignment, E the input projection and D the diagonal
slot occupancy. Node drive is S E x and pooling is D^-1 S^T. A square
permutation changes S through a spatial permutation shared across channels.
The prototype always uses a bijection, including during proposal evaluation;
it does not train a soft dense map and replace it with a different hard map
at inference.

Two controls are mathematical requirements:

1. A global channel permutation can be absorbed by the existing learned input
   and output 1x1 projections. Learning only that permutation adds no new
   interface expressivity.
2. Simultaneously relabeling graph nodes, node biases and node assignments
   preserves every output. A graph obtained only by node relabeling is not an
   independent biological-topology control.

A general square permutation cannot be absorbed by a shared pointwise channel
projection. It changes which board locations communicate through the fixed
graph. This makes it a specific test of the spatial-assignment hypothesis.

Unrestricted learned dense input and output maps pose a different problem.
Even with all graph messages removed, node biases and nonlinearities yield
a whole-board nonlinear network through those maps. Its gains could be
independent of graph communication. The hard spatial permutation avoids that
particular bypass, although node-local controls are still necessary.

## Candidate comparison, not yet launched

After a fixed warmup, compare one prescribed square-swap proposal with the
current assignment on the same training minibatch at fixed weights. The
learned-map arm accepts a strict finite loss decrease; its fixed-map control
executes both forward passes but retains its current assignment. Both then
receive the same ordinary optimizer update. Proposal generation, warmup,
acceptance criterion and budget must be frozen before training. No development
labels, game outcomes or engine evaluation may choose a map.

The minimal topology interaction compares biology and each of the three
degree-preserving rewires with fixed and learned maps across three paired
seeds: 24 fits. Fixed and learned node-local controls add six fits. Every graph
optimizes its own mapping under the same budget. Transferring a map optimized
only for biology to the controls would privilege biology.

For every rewire r, measure the difference between biology's fixed-to-learned
loss improvement and the rewire's corresponding improvement. A biological
mapping explanation needs positive interactions and better final quality than
every learned-map rewire on ordinary and shifted conditions. A gain shared by
rewires or node-local models supports a generic interface effect. Every
proposal, forward call and complete decision must count toward compute.

The exact training protocol, useful margin, backbone and evaluation inputs
remain unselected. The live outcome-supervision study should inform the shared
reference before another training campaign starts. A successful mapping trial
would still need stronger-engine grading, actual gameplay and new confirmation
games before an architectural advantage claim.

## Closest checked prior work

- [FLYNN, Methods II-B](https://arxiv.org/html/2607.00025v1) uses anatomical
  visual sampling and learned input/output MLPs around fly-derived connectivity.
- [Joint optimization of reservoir input and readout](https://arxiv.org/abs/2604.20519)
  already studies trained, constrained coupling to a fixed recurrent system.
- [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) establishes
  differentiable spatial sampling. A coordinate-based variant would need to
  distinguish its contribution from that established mechanism.
- [conn2res](https://www.nature.com/articles/s41467-024-44900-4) treats input
  and readout node selection as explicit choices in connectome reservoirs.

These sources motivate a controlled interface experiment. They preclude a
broad novelty claim from merely learning input and output connections. This
is targeted prior-work checking, not proof that the hard-permutation variant
is new.
