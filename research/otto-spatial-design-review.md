# Spatial value readout: concrete design for review

**Proposal only, unimplemented and unexecuted.** This document refines the [outcome-blind architecture review](../output/otto-conditioning-control-v1/next-architecture-review.md). It uses closed historical reports and inspected source, not partial conditioning-control outcomes. It freezes no future experiment, changes no historical source and authorizes no training or simulation.

The hypothesis is specific: nonlinear interactions between nearby probability masses, computed before pooling, may give better successor-value ordering than compressing a belief directly into a few linear projections. The required outcome is competent autonomous control at an acceptable complete cost. Lower Monte Carlo MSE, more parameters, spatial-looking weights or a biological analogy are insufficient.

## Common public input and lossless geometry

Retain the qualified [explicit branch operator](../src/openjev/research/otto_value_branches.py). For each action `a` and nonfound hit `h`, it constructs joint mass `u`, raw mass `r=sum(u)`, weight `w=max(r,1e-10)` and centered input `Z=u/w`. There are sixteen float64 `105x105` inputs, including blocked stay-and-observe actions. Every model receives the same inputs, successor position `q=(qx,qy)` and supplied sensing length `lambda`. There is no hidden source, prior action history, remaining horizon or new sensor input.

The physical `53x53` field is an exact slice:

`z[x,y] = Z[52-qx+x, 52-qy+y]`, for `x,y=0,...,52`.

Before accepting an input, validate integer in-board `q`, finite nonnegative values, the inherited mass bound and exact numerical zero outside that rectangle. Recenter the slice and require elementwise equality to `Z`; reject, rather than discard, nonzero padding. This is an information-preserving representation change. The mass `m` is the common canonical row-major sum of the original centered array, using the declared arithmetic dtype; all five heads use that same reduction for the skip and context. Do not silently change reduction order per architecture.

All controls have access to the same physical position, relative coordinates, board geometry and known sensing length, either as deterministic input channels or the equivalent complete grid plus position. No learned feature receives the evaluator's state. Source-cell coordinates are grid indices, not a revealed source location.

## Exact proposed spatial head

For odd `k` in `{3,9,27}`, define an unnormalized local mass sum

`B_k(z)[x,y] = sum z[i,j]` over `|i-x|,|j-y| <= (k-1)/2` inside the board.

Values outside the board are zero. Do not divide by window area, renormalize near boundaries, wrap, reflect or periodically pad. A clipped window contains less possible source support; the model receives that geometry explicitly. Prefix-sum or separable implementations must be independently compared with the declared direct sums before use, including rounding near action ties.

At each source cell construct twelve channels, in this fixed order:

`v = [53*z, B_3(z), B_9(z), B_27(z), (x-qx)/52, (y-qy)/52, x/52, (52-x)/52, y/52, (52-y)/52, lambda/5, m]`.

Apply a shared pointwise ReLU MLP `phi:12->24->8`, with biases in both layers. Pool **after** these nonlinearities:

`s = sum_(x,y) z[x,y] * phi(v[x,y])`.

The scalar head is `rho:12->16->1`, ReLU only in the hidden layer, with biases in both layers. Its input is `[s, m, m*qx/52, m*qy/52, m*lambda/5]`. The normalized value is

`f(Z,q,lambda) = c0*m + rho([s, m, m*qx/52, m*qy/52, m*lambda/5])`.

The final layer is signed and unconstrained. `c0` is the same TRAIN-only baseline for every head. This candidate has **737 trainable parameters**; the fixed baseline is not one of them. The gain 53 is a prospective common raw-density scaling choice, not selected from the active autonomous experiment. Local sums remain raw probability mass.

This is not a concave or positively homogeneous network. Scaling `z` changes its local inputs and activations; no algebraic branch fusion is permitted. At `Z=0`, pooling and mass-scaled context vanish, but `rho(0)` may be nonzero after fitting. Preserve it. All controls likewise allow biased zero-input values; do not add candidate-only terminal or zero-value repairs. Subfloor branches remain subnormalized, with `m=r/w`, rather than being rescaled to unit mass.

Physical prediction is `64*f`. Every learned arm retains all four raw costs `1+sum_h w[a,h]*64*f[a,h]`, the unchanged in-bounds selection and strict first-ID `1e-10` tie rule. The positive floor also weights biased zero-input predictions. No clipping or analytic fallback is part of learned deployment.

## Five necessary learned families

These are fresh controls on identical data and targets, not comparisons with selectively chosen historical checkpoints. All use the same raw `c0*m` skip, mass-scaled global context, signed output and public information.

| Family | Exact role and suggested implementation |
| --- | --- |
| Spatial sums | The 737-parameter head above. Tests nonlinear neighborhood information before compression. |
| Neighbor-free | Identical parameter shapes and paired initialization, with separately trained parameters; same geometry, pooling and readout. Replace each of the three `B_k(z)` channels by `z[x,y]`. This retains the nominal 737 parameters but reduces effective input rank, intentionally. It isolates access to neighbors, not every possible conditioning effect: box-sum amplitudes also differ from point mass. Report that limitation; a later scale-matched ablation would be separate. |
| Ordinary CNN | On `53*z`, apply three biased `3x3` convolutions, channels `1->4->4->4`, dilations `1,3,9`, with ReLU after each. Their receptive fields are `3,9,27`. Retain the physical board only after each layer and zero-pad the next; do not propagate activations through out-of-board cells. Concatenate the three four-channel levels with the same eight geometry/context channels used by the spatial head. Apply pointwise `20->8->8` ReLU layers, probability-weighted pooling and the same `rho`. Total **801 parameters**. This tests whether ordinary learned convolution explains the result. |
| Ordinary dense | Fresh `11028->128->1` ReLU model: flattened `53*Z` plus the same three mass-scaled position/lambda inputs. Use the common mass reduction for its skip. All local geometry is derivable from this complete grid and position. Total **1,411,841 parameters**, with roughly comparable matrix work to the spatial candidate, not comparable parameter count. |
| Familiar statistics | A `12->16->1` ReLU head, **225 parameters**, over the twelve branch statistics specified below. This tests whether expensive local processing improves on cheap summaries containing the analytic controller's main ingredients. |

The statistics vector is `[m, m*lambda/5, m*qx/52, m*qy/52, H/log2(2809), sum(z*abs(dx))/52, sum(z*abs(dy))/52, sum(z*dx^2)/52^2, sum(z*dy^2)/52^2, sum(z*dx*dy)/52^2, sum(z^2), max(z)]`, with `dx=x-qx`, `dy=y-qy`. Use `H=-sum(z*log2(z))` only where `z>1e-10`, matching the existing analytic entropy convention. Do not normalize `z` for these summaries. All features vanish at zero input; a learned output bias still survives.

**Same public information does not mean identical features.** Dense128 uses raw centered mass and three context values; the ordinary CNN learns spatial features from raw mass. Neither receives the candidate's twelve-channel tensor unchanged. They are work-comparable alternative representations, so beating them would support the complete feature/aggregation recipe, not isolate a particular neural operator. Equal receptive-field sizes also do not imply equal functions.

For the identical-feature check, an ordinary `1x1` CNN with channels `12->24->8`, the same biases, activations, probability-weighted pooling and `rho` is algebraically this candidate itself. Weight-translation and numerical parity should be engineering fixtures; training it twice would not create independent architectural evidence. A claim about a different operator on these features would require a genuinely different identical-input control, such as a flattened dense head or learned local CNN on the full twelve channels. Its dimensions, zero-input context semantics and full preprocessing cost would need their own prospective comparison. A width-128 dense layer on that full tensor is substantially larger than the raw Dense128 listed here. This proposal makes no such narrow operator claim.

The unchanged [space-aware analytic controller](../src/openjev/research/otto_reference_control.py) is mandatory as a sixth, untrained control. Its branch objective depends on entropy and expected Manhattan distance. The statistics network receives those ingredients, but its MC-fitted output is not asserted to reproduce the analytic formula or its small-mass arithmetic.

For a clean proposed initialization, zero the final output weights and bias in every learned family, with ordinary local-generator He initialization of hidden layers. Every fresh head then begins at the same `c0*m` function. Hidden layers receive zero gradient on the first update and can receive gradients after the output weights change; qualification must test this explicitly. Seeds pair orders and initial functions, not differently shaped hidden tensors. This is a new common initialization, so old trained checkpoints remain descriptive references only.

## Geometry and boundary claims

The shared cell transform is invariant to the order in which complete cell tuples are summed in exact arithmetic. It is not invariant to relocating probability between cells. The proposed signed coordinates and unconstrained weights do not enforce D4 symmetry. Translating agent and belief inside a fixed finite board changes boundary relationships, so whole-policy translation invariance would generally be wrong. No augmentation or symmetry ensemble is added to this comparison.

At an edge or corner, local sums use actual board support and CNN padding is always zero. A visited or zero-probability cell inside the board is distinct from outside-board padding: its intermediate CNN features may carry neighboring information, while its final probability weight is zero. Candidate and controls retain the actor's exact public posterior; they do not invent a new permanent mask or recurrent memory.

## Source-derived work and memory estimates

The following count matrix multiply-accumulates for one sixteen-branch decision. They exclude local sums, biases, activations, pooling, entropy, features, allocations and posterior updates, and are **not latency measurements**.

| Head | Trainable parameters | Matrix MACs per decision |
| --- | ---: | ---: |
| Spatial / neighbor-free | 737 | 21,576,448 |
| Ordinary CNN | 801 | 24,632,640 |
| Dense128 | 1,411,841 | 22,587,392 |
| Statistics | 225 | 3,328, plus statistic construction |

Thus the spatial/CNN comparison is approximately parameter- and work-matched; dense128 is approximately matrix-work-matched only. Neither equality of updates nor these counts establishes equal elapsed compute. Full source-bound timing must include unpacking, validation, box sums/convolutions, setup, all branch values and final public updates.

Sixteen float64 centered arrays occupy **1,411,200 bytes**; their physical slices contain **359,552 bytes**. The candidate's 24-channel hidden tensor alone occupies **8,629,248 bytes** at deployment. A float32 training batch of 128 needs **34,516,992 bytes** for that layer alone, before other activations, gradients, optimizer state and frameworks. Evaluating the same local head over all padded `105x105` cells would require about **84.7 million matrix MACs**, nearly four times the physical-grid work. Lossless slicing is therefore material, but is not a measured acceleration.

Naive `27x27` box loops and convolution `im2col` buffers can dominate CPU time and memory. Geometry tables may be derived from public positions and reused, but their construction/storage must be charged. Do not retain expanded per-cell TRAIN features: one 80-epoch candidate fit already entails approximately **603 billion forward matrix MACs**, before backpropagation. Chunked feature construction and a separately bounded synthetic runtime qualification are necessary before committing to fifteen fits. No optimizer batch size, epoch count or architecture should be changed after scientific results to fit a budget.

## Training, competence and falsification

The clean first scientific comparison would use all five families at three paired seeds, the same 5,589 cached TRAIN prefixes and uniform MC target `(T-t)/64`, fixed final checkpoints and exposed 1,109-row VALID diagnostics. Eighty epochs/batch128 would imply **52,800 optimizer updates across fifteen fits**, but that workload is a proposal pending runtime qualification, not an admitted allocation. Choose any reduced common schedule before new outcome reads, not a different schedule for a struggling arm. A new loss, per-model Bellman targets, extra state collection, teacher action labels or remaining-time input would change the question and must not be added only to the candidate.

The strongest risk is that this shared MC objective still fails to teach action ranking. Historical min8, capacity and Bellman findings do not prove that geometry is the missing cause. A familiar-statistics head may be sufficient, or it may fail to learn the cheap analytic objective despite receiving its ingredients. That outcome would argue against escalating spatial complexity on these labels; it would motivate a separately matched objective/coverage test, not a claim that more recurrence is required.

Scalar MSE must remain diagnostic. A prospective autonomous decision must retain an absolute anchor such as at least 95% weighted success and capped moves no greater than 1.05 times analytic control for **every candidate seed and setting**, plus meaningful paired gains over ordinary controls and complete controller cost. Exact improvement/cost thresholds and fresh cohorts require a later frozen protocol. Do not promote a winner from exposed validation or pool away a failed setting. All failures count the full horizon. The current study and all historical failed gates remain unchanged.

Synthetic qualification should cover lossless unpack/recenter at all board boundaries, direct-sum local kernels, zero/subfloor/point/diffuse beliefs, no outside-board convolution propagation, every layer's gradients, immutable checkpoint/float64 deployment parity and strict action ties. A constructed pair distinguishable only through neighborhoods can establish representational capability; without demonstrated reachability and improved autonomous decisions it establishes no task benefit.

## Prior art and claim limit

[PBVI, sections 2-3](https://robots.stanford.edu/papers/Pineau03a.pdf) connects belief-space value planes, actions and reachable-belief coverage. Its optimal-value structure does not certify concavity of realized returns from our belief-adaptive heuristic teacher. [Value Iteration Networks, sections 3-4](https://proceedings.neurips.cc/paper/2016/file/c21002f464c5fc5bee3b98ced83963b8-Paper.pdf) establishes structured spatial planning computation versus ordinary convolution. This proposal is not value iteration over the full belief state. [Deep Sets, section 3](https://proceedings.neurips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf) establishes shared nonlinear transforms followed by pooling. These are mechanisms to compare, not evidence of novelty or olfactory efficacy.

A connectome mask is not part of this experiment. It would require a meaningful unit mapping and degree-preserving rewired, random-sparse, ordinary dense and task-local controls with measured work. With exact public filtering already provided, neither a biological mask nor recurrence resolves the demonstrated lack of learned controller competence by itself.
