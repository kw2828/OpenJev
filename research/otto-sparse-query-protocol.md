# Sparse-query control opportunity test

Prospective protocol. Freeze sources, tests, successful engineering evidence,
native/runtime/model inputs and the seed review before collecting either stage.
The V2 local-advantage recipe remains closed after its 7/11 failure. This study
does not fit those labels, train a model, or change any existing result.

## Question and controls

Can sparse calls to the existing neural planner preserve useful search behavior
while lowering total controller computation? **Period-two querying is the sole
primary candidate.** Random-pair and entropy gates are diagnostic controls, not
replacement winners if the primary candidate fails. The standalone original
restricted neural planner and analytic policy are the two references.

Use the qualified 53-by-53 OTTO environment, public filter, original float32
neural tie selection with eight-way symmetry averaging, and float64 analytic
endpoint. Sparse actors use the unchanged QueryGateActor and its 31 public
features. Only actual queries can invoke the neural scorer. No skipped-state
annotations, hidden native source, future observations or evaluator outcomes
may enter gate features. No normalization repair of inherited public beliefs.

For one-based decision count t, every sparse arm must obey the causal ceiling
`Q_t <= ceil(t/2)`, with counts reset at each episode. Skipping selects the
current analytic action, not the previous neural action. Querying selects the
current neural action. The shared ceiling is not a guarantee of equal actual
query counts or computation.

- **Period-two:** query at preaction steps 0, 2, 4, and so on.
- **Random-pair:** one query position per two-decision block. The fixed helper
  derives a deterministic bit from its separate SHA256 namespace,
  episode seed and pair index. The bit selects offset zero or one, so every
  completed pair contains exactly one query. An odd final prefix can differ
  from period-two by one call. Preserve all bit witnesses.
- **Entropy:** query if the exact inherited float32 entropy feature (index 19)
  is at least the frozen threshold and unused causal credit remains. It can
  underspend the ceiling. There is no forced first query or threshold search.

The sparse state is two exact float32 counters: preaction step and query count.
This is a deterministic accounting state, not a learned recurrent model. Gate
inputs and returned states are detached, finite and immutable. The encoded
step must match the chronological counter exactly under the original float32
feature encoding.

## VALID and single calibration rule

Collect 24 complete period-two VALID paths: twelve cases each at sensing
lengths 3 and 4, seeds 20100001-20100012 and 20200001-20200012. Initial hit is
`1 + case_index % 3`. Retain every preaction float32 entropy value, action,
query, score witness, observation, posterior hash, timing and final update.
End only at discovery or 2,188 moves. No replacements or shortened tails.

Give every episode total weight 1/24, divided equally among its actual preaction
rows. Sort by entropy, episode index and step. Select the smallest entropy
whose cumulative exact rational weight reaches at least 1/2. This is one pooled
lower weighted median with fixed high-entropy direction; no reward, success,
search-length optimization, threshold sweep or selection among quantiles.
Episode length determines row weights only. Do not calibrate length 5.

All 24 VALID paths and journals must close and the threshold record must be
durably published before the first EVAL reset. Deploy that same threshold in
all three EVAL settings. VALID paths are calibration data, not evaluation.

## Fresh autonomous evaluation

Collect 24 paired cases at each sensing length 3, 4 and 5, with seeds
20300001-20300024, 20400001-20400024 and 20500001-20500024. The seed review must
find no prior use within its declared text/evidence scope before freezing.
Use `initial_hit = 1 + case % 3` and `block = case // 3`, yielding eight
hit-balanced blocks per setting. The five-arm base order is analytic, neural,
period2, random_pair, entropy. Rotate it left by global case index modulo five.
This gives 360 EVAL episodes after the 24 VALID paths.

Native source and indexed observation-uniform streams are paired across arms
within each case. Their actual observations may diverge with their paths.
Preserve all 384 paths, failures, slow episodes and capped tails, including the
public update on discovery or on move 2,188. There is no early promotion or
replacement. Length 5 is a separately reported, previously known type of
shift, not a newly discovered robustness environment.

Use the qualified environment's positive-initial-hit mixture to report weighted
success, capped moves, queries and costs by arm/setting and all eight blocks.
Also retain raw totals and every episode. Never average only successes or
querying states. Query counts include odd final prefixes; actual neural calls
must equal deployed queries with zero annotations.

## Costs and the sixteen-condition rule

Online controller time includes construction, filtering, analytic scoring,
feature construction, gate decisions, neural calls and allocated fresh setup.
Standalone references avoid gate work they do not require. Measured journal
serialization/I/O is excluded from controller timers; any other monitoring
remaining inside an operation stays charged. Preserve nested timers without
adding overlapping intervals, and report whole-process time separately.

Allocate fresh common setup over all 384 paths, TensorFlow/model setup over
312 neural-capable paths, and gate-module setup over 240 sparse paths. These
counts include VALID. No inherited 192/576-episode setup allocation is reused.
Record first-inference cost and no uncounted warmup calls.

The calibration bill is the full physical VALID collection interval plus
threshold construction and durable publication, plus the 24 VALID shares of
the three setup categories above. It includes that stage's simulation and
journaling expense; do not add its nested decision timers again. Charge the
bill exactly once, divided over all 72 entropy EVAL episodes, including length
5. Report online cost and this fully paid controller cost separately. Period2,
random-pair and the references have no calibration charge. Evaluation-only
verification and whole-process lifecycle overhead remain separately visible.

All **16 primary conditions** must pass:

1. All 24 VALID and 360 EVAL paths complete, original process closes within its
   limit, and the independent saved-record audit agrees (one condition).
2. Every sparse prefix obeys its causal quota (one condition).
3. Both reference arms reach at least 95% weighted success at each primary
   length 3 and 4 (four conditions).
4. At each primary length, period2 has success at least 95%; success no lower
   than either reference; capped moves at most 105% of neural; capped moves at
   most 95% of analytic; and fully paid controller time at most 60% of neural
   (ten conditions).

Report the same five candidate comparisons for all three sparse arms and all
three lengths (45 diagnostic conditions), plus the earlier 50%-cost comparison
in each such cell (nine further diagnostics). These 54 diagnostics cannot
substitute for a failed primary condition. The new 60% bound allows overhead
beyond a half-query schedule while requiring at least 40% total cost reduction;
it does not retroactively relax or reverse any prior 50%-cost failure.

A pass establishes a sparse-control opportunity worthy of a separate learned
allocation study, not a recurrence, world-model, biological-wiring, RL or
architecture advantage. A failure closes this primary recipe under this
allocation. Do not retune thresholds, replace the primary arm, increase the
budget or retry on different cases. Any later study needs a new mechanism,
protocol and untouched evaluation cohort. Existing exposed cases stay exposed.

## Resources and independent evidence

One original suspend-inclusive supervisor, 1,800 seconds, one numerical CPU
thread, unchanged qualified native interpreter, 4 GiB RSS and 6 GiB output.
This prospective two-stage cap is not a completion guarantee or an extension of
any prior allocation. Maximum 384 resets, 840,192 actual moves, 420,096 neural
calls. There are zero optimizer updates, local-advantage samples or teacher
annotations. Retain attempted/returned counts and pending operations on failure.

Use exclusive output paths and the previously qualified compressed,
episode-durable ledger contract: calls acknowledge encoded events within a
pending episode; a completed episode acknowledges all its journals only after
flush/fsync, followed by its durable completion row. Preserve the episode-start
record and partially written journals if interrupted. No resume or silent retry.

Freeze independent audit code before the scientific run. Its separate original
supervisor allows 120 seconds, 2 GiB RSS and 128 MiB output. It authenticates all
saved inputs, source hashes, original process identity/closure and payloads,
then independently reconstructs chronology, schedule bits, quota counters,
threshold selection, saved-score action selection, paired random-stream
witnesses, complete costs and every condition. It makes no new model, native,
filter, sampler, optimizer or numerical-array calls. Numerical belief filtering
and neural values remain inherited qualified producer evidence. Publish the
complete result even if the continuation rule fails.
