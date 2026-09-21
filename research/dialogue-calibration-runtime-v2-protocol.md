# A separate runtime qualification for the unchanged temperature control

21 September 2026. This is a new resource study, following the completed V1
replay and its failed admission estimate. V1 projected 5,081.543 seconds against
an 1,800-second limit. Its maximum paid rate came from the first small case;
startup, shape effects and timing noise have not been causally separated. That
failure is retained, and its inference command remains unused. The new study
must be frozen before selecting its timing cohort or making its model calls.

The scientific experiment remains the same [512-dialogue temperature
control](dialogue-calibration-control-protocol.md): the authenticated TRAIN
complement, all twelve final checkpoints, unchanged full-dialogue inference,
one scalar per checkpoint fit only on calibration data, and all eleven quality
conditions. Original V2 remains a raw failure at six of seven conditions. No
architecture improvement or untouched confirmation follows from this pilot.

## Fixed workload and prospective verification

From the original 2,363 evaluated DEV workload profiles, exclude the two V1
replay IDs. Sort remaining IDs by `SHA256("openjev-calibration-runtime-v2:" +
dialogue_id)`, breaking hash ties by ID. Choose exactly 128: first 64 estimator,
next 64 verification. Retain this order for every checkpoint. No labels,
predictions, correctness, latency or task scores enter selection. DEV has
already been exposed by the earlier scientific comparison and diagnostic.

For each of the twelve checkpoints in original seed-outer order, load once,
replay the two legacy cases as paid warmups, then execute the entire estimator
block. Persist a verification-time prediction before any verification forward.
Then execute the entire verification block, without changing that prediction.
All 130 complete public streams per checkpoint use the immutable V1 restoration
and inference wrapper. Preserve raw endpoint arrays and per-dialogue journals.
Every replay must match the saved original DEV output under unchanged support,
float32, mass, first-argmax and numerical tolerances. Monitor every new recurrent
update and confirm encoder and memory weights stayed unchanged. This does not
establish equivalence of old hidden trajectories that were never saved.

Let E_f be the paid synchronized estimator-block duration for fit f. Define
W(S) using four totals: dialogue count, encoder calls, padded attention
positions, and real question updates. The pre-verification forecast is

`V_f = 2 * E_f * max_k(W(verification)[k] / W(estimator)[k])`.

Require actual paid verification duration <= V_f for **every** fit. No pooling
away failures, changing cohorts, dropping the first fit, trimming outliers or
fitting timing coefficients from the verification block is allowed. Continue
the fixed pilot to record all twelve results even if a completed block fails
its timing forecast, provided resource and parity checks still hold.

The full 512-dialogue calibration estimate is

`sum_f(2 * E_f * max_k(W(calibration)[k] / W(estimator)[k]))`

plus all twelve observed loading times, all 24 warmup costs, the completed
preparation's actual parent wall time, and measured pilot non-block overhead.
The overhead snapshot occurs after final upstream authentication and initial
payload hashing; it excludes the disjoint load, warmup, estimator and
verification intervals. Projection and receipt publication follow the snapshot.
Their residual duration must be exposed separately by the actual parent exit.
The complete pilot cost is also reported separately, including all estimator
and verification computation. Do not describe its sunk cost as free.

Admission requires all twelve timing checks and total estimated cost <= the
unchanged **1,800 seconds**. The factor of two applies both to verification
and full inference. It is a fixed margin, not a fitted confidence bound. These
work proxies and block checks do not prove linear scaling or future runtime.
A diagnostic reports saved geometry ranges and calibration cases outside the
measured ranges; actual joint batch-shape coverage is unknown from these saved
profiles. Geometry is not an additional admission criterion or a license to
reselect timing cases. The separately enforced deadline remains the protection
against projection error.

## Authentication, budgets and outcomes

The prospective source closure contains this protocol plus the common
controller, pure projection, geometry diagnostic, pilot and inference runners,
and their tests. Qualify with synthetic inputs only, then freeze source hashes,
external pins, runtime versions, fixed outputs and qualification evidence.
Authenticate the complete original source/data/model closure and all thirteen
V1 control sources, preparation and qualification receipts and actual parent
exits before reading task records or loading weights. Reauthenticate at exit.

The pilot has 600 native suspend-inclusive seconds, 8 GiB RSS, 8 GiB sampled MPS
driver memory and 512 MiB output. Its success payload is started/sample/projection
JSON plus twelve predictions/journal/forecast/completion groups and the root
completion. A process failure preserves partial arrays, progress and an error
receipt. A completed but denied projection is a valid negative resource result.
No automatic retry, changed sample, checkpoint replacement or budget extension.

Only a complete pilot with a successful actual parent exit and recomputed
admission can authorize the new study's inference phase. That phase has
**3,600** suspend-inclusive seconds, 8 GiB RSS, 8 GiB sampled MPS memory and
512 MiB output. Restore the same twelve checkpoints and execute the same 512
calibration dialogues per fit, with exactly the original public stream,
candidates, chunking/pooling, lexical variant, autonomous state and numerical
route. Save raw float32 endpoint probabilities with canonical row identities.
All 6,144 forwards must complete. Charge all loads, preparation of inputs,
transfers, synchronization, writes and cleanup. No optimizer, temperature
application, task-metric calculation or official TEST access is allowed here.

The original V1 inference directory remains unused. New outputs are exclusive
under `runs/dialogue-calibration-runtime-v2/`. Any subsequent scalar fitting and
reporting needs a separately qualified reader bound to these actual completions
and parent exits; it must retain the original eleven scientific conditions and
report the old raw and resource failures alongside any new result. Resource
admission alone is neither a calibration result nor a novel-architecture claim.
