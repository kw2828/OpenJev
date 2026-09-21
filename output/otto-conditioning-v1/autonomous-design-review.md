# Proposed autonomous conditioning comparison

**Unexecuted design recommendation, not an admitted plan.** This note was written from existing protocols and seed-review metadata without reading conditioning qualification/full-study outcomes, arrays or checkpoints. No model, training or simulator call was made. The [scalar protocol](../../research/otto-conditioning-protocol.md) requires this comparison after technical completion and independent agreement, regardless of its six scalar conditions. Freeze implementation, numerical qualification, exact input/source/runtime identities and the autonomous plan separately before execution.

## Smallest useful fixed cohort

Retain all six fixed final conditioning checkpoints: gains 1/53 at seeds 10101, 10102 and 10103. Add the unchanged qualified `analytic_inbounds` controller. No seed selection, additional training, calibration, clipping or fallback is introduced.

Recommend **24 cases per setting**, eight blocks of three cases, with `initial_hit=1+(case%3)` and `block=case//3`. This is the smallest allocation retaining the existing eight-block comparison while representing all three initial-hit strata in every block. It gives **72 cases and 504 episodes** across seven arms. Each learned head runs all 72 cases. Three fitting seeds do not turn one environmental case into three independent cases; report all paired fits and acknowledge the small environmental sample.

| Setting | Proposed episode seeds, inclusive | Interpretation |
|---|---|---|
| lambda3 | 15100001-15100024 | Training-supported sensing length, fresh cases |
| lambda4 | 15200001-15200024 | Training-supported sensing length, fresh cases |
| lambda5 | 15300001-15300024 | Unseen sensing length with its exact kernel supplied |

Use new template seeds 15500001-15500003. These ranges are absent from the recorded ranges in the [existing ledger review](../otto-coverage-v1/seed-ledger-review-01.json) and from a source/protocol token search made for this note. The ledger predates the failed coverage collection; its actual collection range and proposed future EVAL ranges are separately declared in the [coverage protocol](../../research/otto-coverage-protocol.md). Leave the reserved starts 14100001/14200001/14300001 untouched. These checks support a proposal, not a refreshed exhaustive nonreuse certificate: repeat the source/plan and actual episode-ledger checks immediately before freezing. Absence is certified only for the inspected records; keyword absence is not proof of global freshness or external-model corpus independence.

Order arms seed-outer, gain 1 then 53, followed by analytic; rotate left by global case index modulo seven. Pair sources and random uniforms by channel/index within each case across arms. Different paths legitimately produce different observations. The three regimes use distinct cases, so cross-regime differences are not paired interventions on the same source realization.

Use the unchanged 53x53 task, four hit categories, `R_dt=2`, Euclidean sensing and horizon **2,188**. End only on finding the source or the horizon; retain the final public update in either case. Require exact actor/native public-belief agreement at every reset and update, using native belief only as an evaluator-side witness. Failures contribute 2,188 capped moves. No stuck-based early stop or replacement episode is allowed. Three template resets verify the already-qualified kernel and initial-hit-mixture identities. Identical native/filter code does not require another native qualification cohort; any material adapter change would require a separately bounded qualification before this plan.

## Transfer definition and deployment checks

Lambda5 changes the observation law and supplied sensing-length feature while retaining grid, action space, source-search objective and native model family. Each actor receives the correct lambda-specific kernel and exact public filtering. This is supplied-model parameter transfer, not unknown-kernel adaptation, spatial-size transfer, a new memory task or broad out-of-distribution robustness. No fitting or choice of policy uses lambda5 trajectories.

Deploy exact float64 upcasts of the fixed float32 checkpoints. Apply the gain only to spatial readout inputs, with raw mass for `c0` and unchanged mass-scaled context. Use all sixteen explicit observation branches, raw masses, `max(mass,1e-10)` weights and signed physical values. Score all four actions; choose the first in-bounds numeric action strictly within `1e-10` of the eligible minimum. Preserve biased zero-input values and subnormalized branches.

Before autonomy, qualify all six checkpoints' scalar branch values, four costs and selected actions against Torch-double references on the same fixed first eight TRAIN and eight VALID states used by the earlier scalar protocol: absolute/relative tolerance `1e-10`, exact eligible action agreement. These 16 mechanical states are not representative validation. Retain source-level zero/subfloor/boundary fixtures and no output repair. Numerical failure stops before EVAL without relaxing tolerances. Scalar pairing at initialization alone is insufficient for this deployed-path check.

## Suggested 30-condition control screen

Use each setting's authenticated initial-hit mixture: first average within hit stratum, then mix. Family means equally average the three fitting seeds. Each block uses its three hit-stratum cases and averages paired fitting seeds. Report every arm, setting, stratum, block, found count, capped-move distribution and failed case. Keep lambda3/4 and lambda5 visible separately; do not pool away transfer failure.

Propose gain53 as the fixed candidate, with all **30** conditions required:

- **18 competence conditions:** each candidate seed in each setting has weighted success at least 0.95 and capped moves at most 1.05 times analytic control.
- **12 paired conditioning conditions:** in each setting the candidate family has no lower success than gain1, capped moves at most 0.95 times gain1, strictly positive paired move gains in at least six of eight blocks, and no greater complete controller seconds per search.

Compute direct unrounded inequalities without epsilon. Also publish the same absolute competence measurements for every gain1 seed. These practical pilot thresholds retain the previous absolute anchor and relative-gain definitions; they are not power-calibrated significance tests or confidence bounds. With only eight cases per stratum, an apparent pass would motivate a larger separately frozen confirmation, not establish a population success guarantee. A relative improvement between incompetent families fails the overall rule. Scalar MSE must neither select checkpoints nor determine whether this screen is executed.

## Complete costs, bounds and audit

Proposed worker caps are **5,400 suspend-inclusive seconds, 8 GiB RSS, 6 GiB output**, **507 resets** and **1,102,752 native steps**. There are 504 evaluation resets plus three templates. At most 945,216 learned decision forwards occur across 432 learned episodes, with sixteen scalar branch rows per forward; mechanical parity work is counted separately. No optimizer updates or new collection are included. These are hard caps, not a prediction that every run will finish. Use a separate saved-output audit capped at 1,800 seconds, 4 GiB RSS and 128 MiB output, with no simulator replay.

Physically load six independent immutable heads once. Allocate each load over its **72** episodes and learned module setup over **432** learned episodes. Charge actor initialization, every branch/copy, gain transform, feature construction, scalar prediction, masking/selection and every public update, including the last. Include inherited analytic-table construction if still incurred by the learned adapter. Do not fold away the scaling or omit its cost after training. Exclude only measured nested artifact I/O; report raw instrumented time, excluded I/O, native reset/step time and complete worker time separately. Retain every first measured inference and do not drop outliers. One rotated run is a hardware-specific measurement, not a repeated latency benchmark.

Report the already-paid six fitting costs separately, without rerunning training. For descriptive amortization at `H=1,100,10000`, use per-head steady controller cost plus `(its fit cost + common fitting preparation/6 + its load + learned shared setup/6)/H`; report original physical totals and separately labeled diagnostics/engineering costs. These accounting scenarios are not an additional scientific gate or simulated deployment demand.

Save every reset/transition, public posterior witness, branch raw masses/floors/values, four raw costs, mask, selected action, evaluator-only source/draw evidence and complete operation/cost ledger. The independent reader must authenticate the completed conditioning study and its audit, then require successful matching autonomous worker/parent closure before decoding outcomes. It must reconstruct all 504 public paths and saved learned readouts and recompute all 30 conditions. Native randomness, optimizer history and timing truth remain authenticated execution evidence. No results, recurrence/connectome advantage or novelty are claimed by this proposal.
