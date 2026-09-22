# Autonomous control with five fixed spatial readouts

This prospective protocol implements the earlier [outcome-blind design](otto-spatial-control-design.md). Qualification and full evaluation are separate modes, each requiring a reviewed implementation, frozen source/input/runtime plan, exclusive output, and original supervised completion. Full evaluation additionally requires successful qualification and its separately counted saved-output audit. No new training is part of either mode.

The [completed scalar study](otto-spatial-study-results.md) did not favor the spatial candidate over the simpler controls: its mean exposed-VALID MSE was 2.246% above neighbor-free and 4.934% above statistics. Those results do not select or exclude a checkpoint here. Retain all fifteen fixed final heads, using every combination of seeds 10101/10102/10103 and families `spatial`, `neighbor_free`, `cnn`, `dense128`, `statistics`. Add unchanged `analytic_inbounds`. Candidate identity is prospectively `spatial`; ordinary controls and the analytic anchor remain required regardless of the scalar ranking.

## Inputs and source identity

Authenticate the completed fifteen-fit plan, worker receipt, original successful process terminal, independent audit and closed payload/source lineage before decoding any head. Bind exactly the fifteen `final-<family>-<seed>.npz` files from that worker. Their saved float32 parameters and baseline are immutable. Do not substitute synthetic qualification weights, initial checkpoints or an older study's fitted model. No calibration, clipping, retraining, fallback or seed replacement is allowed.

Reuse the pinned 53x53 public posterior, explicit sixteen-branch implementation, native/filter contract and saved float64 observation kernels for sensing lengths 3, 4 and 5. The new source closure is additive; do not change any historical frozen source. The Python/runtime distribution identities and one-thread numerical settings remain those of the completed training study. Qualification selects only the first eight TRAIN and first eight VALID public states from the original authenticated caches, plus fixed mechanical fixtures. Compressed NPZ storage requires decompressing the selected belief, position and sensing-length columns before slicing; neither qualification route accesses target or legacy feature arrays. Full evaluation does not read those caches or rerun parity.

## Separate deployed-branch qualification

Qualification makes no environment reset or step and fits no model. For each head, load an immutable NumPy deployment and restore a Torch64 reference from the same saved float32 parameters. Both routes receive identical public branch inputs and explicit integer successor positions and supplied sensing lengths. Each head sees exactly 52 tuples in the following order:

1. Original TRAIN rows 0 through 7, followed by original VALID rows 0 through 7. Retain each row's original float64 belief, public position, eligible actions, sensing length, regime and metadata/hash joins. Do not use targets or fit diagnostics.
2. Thirty-six synthetic tuples, ordered by sensing length 3/4/5, then positions `center=(26,26)`, `lower=(0,0)`, `upper=(52,52)`, then belief kinds `asymmetric`, `point_successor`, `zero`, `subfloor`.

For each synthetic position, let integer grid indices be `x,y` in 0 through 52. Set `r[x,y] = 1 + ((17*x + 29*y + 7*x*y) % 97)` and set `r[position]=0`. Convert to float64 and define `asymmetric = r / sum(r,dtype=float64)`. The `point_successor` belief has unit mass at the successor of the first numeric in-bounds action: `(25,26)` at center (action 0), `(1,0)` at lower (action 1), and `(51,52)` at upper (action 0). The `zero` belief is all zeros. The `subfloor` belief is the asymmetric belief multiplied by float64 `1e-12`. Determine eligible action IDs from board bounds only. Use the exact saved kernel for the supplied sensing length. These are numerical fixtures, including nonnormalized and impossible public-state limits, not additional environment observations or evidence of task competence.

Construct all sixteen nonfound branches in action-major/hit-minor order. Preserve raw masses `m`, weights `w=max(m,1e-10)`, centered inputs `u/w`, successor positions, signed physical values `64*f(u/w)`, and biased values on zero inputs. Retain blocked stay-and-observe action costs, while excluding blocked IDs from selection. Compute all four costs `1+sum_h(w*value)` and choose the first numeric eligible ID strictly within `1e-10` of the eligible minimum. No weight renormalization, zero-value shortcut or terminal override is allowed.

The same 52 tuples are shared by all heads. Store their exact float64 beliefs, both centered branch views, raw masses, weights and sensing lengths, plus integer positions/successors and eligible masks, once. Both complete centered views occupy 146,764,800 uncompressed bytes. Per-head records reference tuple IDs and hashes; they must not repeat these arrays. Save all sixteen NumPy and Torch64 physical values, four costs, chosen actions, masks and original checkpoint identity for each pair.

Require every finite physical branch value and action cost to satisfy `abs(numpy-reference) <= 1e-8 + 1e-10*abs(reference)`, and require exact selected-action equality. This physical-value absolute term is intentionally 64 times stricter in normalized units than the scalar study's absolute tolerance. It is a separate deployment requirement; a failed comparison cannot cause a relaxed threshold or alternative tie rule.

The allocation is exactly **15 head loads, 15 Torch64 restorations, 780 NumPy forwards and 780 Torch64 forwards**, each forward evaluating sixteen branches. Compare **12,480 paired branch values, 3,120 paired action costs and 780 paired actions**. No additional current-state prediction, training, native call or external model call is admitted. Worker bounds are **600 suspend-inclusive seconds, 4 GiB RSS and 256 MiB output**. The independent qualification audit has a separate allocation of the same size and performs exactly **780 saved NumPy forwards**, not another Torch or native execution. Shared NumPy algebra is disclosed; geometry and comparison arithmetic are reconstructed independently.

## Fresh autonomous cohort and public information

Full mode retains the unchanged 53x53 sampled-source environment, four hit categories, `R_dt=2`, Euclidean sensing and horizon **2,188**. Use these proposed ranges, after the separately pinned scoped seed-ledger review verifies nonoverlap with the inspected local history:

| Setting | Inclusive episode seeds | Cases | Arms | Episodes |
| --- | --- | ---: | ---: | ---: |
| lambda3 | 16100001-16100024 | 24 | 16 | 384 |
| lambda4 | 16200001-16200024 | 24 | 16 | 384 |
| lambda5 | 16300001-16300024 | 24 | 16 | 384 |

For zero-based case `c`, initial hit is `1+c%3` and block is `c//3`. There are eight complete three-stratum blocks per setting and **72 paired environmental cases, producing 1,152 episodes**. Fitting seeds are repeated controllers on the same environmental cases, not additional independent environmental samples.

Use seed-outer/family-inner order with the five families above, followed by analytic control. Rotate the sixteen-arm list left by `(setting_index*24+c)%16`. Pair sampled sources and categorical uniforms by channel/index within a case without forcing the same observations along different paths. The three settings use distinct cases. Template seeds **16500001, 16500002 and 16500003** are three separately counted resets for lambda3/4/5; require exact agreement with the already authenticated kernels and positive-initial-hit mixture weights.

The actor owns its public posterior. Source coordinates, native hidden state, random generator state and evaluator-only draw records must not enter its choice. Exact public/native posterior agreement is required at every reset and update. Retain and assimilate the final found or censored packet and charge that update. End only on finding the source or reaching the full horizon. No stuck stop, trajectory replacement, anti-cycle rule or privileged analytic fallback is introduced.

Supply the corresponding sensing length explicitly to the readout from the task's public configuration. Lambda3/4 are training-supported settings. Lambda5 supplies an unseen parameter and its exact kernel on the same board. It tests supplied-model parameter extrapolation, not adaptation to unknown dynamics, new grid geometry, robotics transfer or a learned world model.

## Fixed sixty-six-condition screen

For each arm and setting, average equally over the eight cases in each initial-hit stratum, then apply the authenticated positive-hit mixture weights. Family means average the three fitting seeds equally. Each block applies the same mixture to its three cases and then averages fitting seeds. Failures contribute all **2,188 capped moves**. Publish every arm/setting cell, stratum, seed, raw success count and block comparison. Do not pool away a failed seed or setting.

All **66** conditions are required, evaluated directly without rounding or an added decision epsilon:

- **18 absolute competence conditions:** for each spatial seed and each setting, weighted success must be at least 0.95 and mean capped moves at most 1.05 times analytic control.
- **48 paired control conditions:** against each of neighbor-free, CNN, dense128 and statistics, separately in each setting, spatial family success must be no lower; capped moves at most 0.95 times the control; strictly positive paired mean move gains must occur in at least 6/8 fixed blocks; and complete controller seconds per search must be no greater.

Publish the identical competence checks for every noncandidate head descriptively, plus the analytic outcomes. No noncandidate is promoted because spatial fails. These are practical small-cohort continuation screens, not calibrated significance tests or population success guarantees. Beating learned controls does not establish an efficiency gain over analytic control; show its measured utility and complete cost explicitly. Even a pass would justify separately frozen confirmation and a second environment, not establish architectural novelty, recurrence, connectome benefit or optimality.

## Complete costs, resource bounds and evidence

Full-worker bounds are **21,600 suspend-inclusive seconds, 8 GiB RSS, 12 GiB output, 1,155 native resets and 2,520,576 native steps**. At most **2,363,040 learned forwards** occur across 1,080 learned episodes, producing 37,808,640 branch values. Head loads are exactly fifteen; full-mode Torch/parity, fitting and external model calls are zero. Successful earlier qualification is consumed through its pinned evidence, not repeated.

These bounds must pass prospective fabricated serialization checks of the final producer and auditor writer schemas before the first qualification plan is frozen. Bind both projections in each plan. Include every operation journal, transition, public/branch witness, duplicated episode/draw record, summary and setup record at maximal allocation. The earlier 4 KiB-per-decision estimate is not itself admission evidence. Increase or revise an output allocation before the first qualification plan is frozen if actual schema projection requires it; never extend a live run or silently discard required records.

Load each head once and allocate its full validation/load cost over its 72 searches. Allocate shared learned-module setup over all 1,080 learned searches. Complete controller cost includes actor initialization and analytic tables where applicable, all branch construction/copies/centering/features, inference, reductions/masking/selection and every public update. Exclude only measured nested artifact I/O and show raw instrumented time and that exclusion separately. Retain first calls and all outliers. Native setup/reset/step time, qualification, earlier fitting, audit and total worker time are separate scopes. One rotated timing pass does not establish repeated latency behavior.

At descriptive amortization horizons `H=1,100,10000`, report controller cost without already allocated deployment setup, plus `(head fit cost + training preparation/15 + head load cost + learned module setup/15)/H`. Do not divide shared setup again per head or charge nested worker/parent time twice. Show original fitting diagnostics and qualification separately. Analytic actor initialization remains paid per search. These views introduce no additional efficacy gate.

Preserve attempted/returned calls, contexts, pending operations, public transitions, raw masses/floors, branch values/costs, masks/actions, posterior witnesses, evaluator-only pairing evidence and original timing records. Check RSS and output limits during operations; those sampled checks are not hard address-space reservations. The original suspend-aware supervisor supplies the hard worker deadline. Require its successful exit, no timeout and absent process group, not just a completion file. Preserve interrupted/failed attempts and partial work. No retry, replacement checkpoint/case, route switch or budget extension is allowed under the same frozen plan.

After successful full original closure, the separately frozen auditor has **21,600 seconds, 4 GiB RSS and 1 GiB output**. It authenticates all source/data/process joins, reconstructs all public trajectories and explicit branch arithmetic, replays every actual learned decision from the saved heads, and checks costs, complete cohort coverage and all 66 conditions. Count every replay forward and preserve compact attempted/returned records. At the maximum, there are 4,726,080 such events; their actual fabricated serialized size and all other audit output must be admitted before freeze. No new environment or optimizer execution occurs.

The saved replay shares the qualified NumPy readout algebra with production. Independent filtering, geometry, arithmetic, metric and journal checks plus the saved Torch64 qualification provide additional evidence. Historical optimization, native randomness and timing truth remain authenticated original execution evidence. Audit agreement verifies the preserved experiment; it cannot turn a failed continuation rule into a positive result. Publication requires successful original completion and independent agreement.
