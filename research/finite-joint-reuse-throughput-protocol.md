# Integrated joint-reuse training throughput

This fresh engineering comparison follows the successful 44-test component
qualification. It neither retries the failed equal-update study nor changes
its 90-second projection gate or 1,024/3,072 update counts. It does not admit
a scientific run, select a model for effectiveness, or evaluate DEV data.

Generate exactly one TRAIN dataset: namespace 943201, split 0, 512 attempted
histories, horizon 2. Retain all attempted histories, endpoint supervision,
generator counts and a separately stored oracle array. The oracle array is
never passed to training. Every fit uses the same tensors and seed 943301,
batch size 64, learning rate 0.003, gradient clip 5, 32 prefix updates and 64
joint updates. Construct a fresh model and fresh optimizers for every fit.
Keep the existing prefix optimizer, prefix-to-joint reset, minibatch schedule,
exact-update controller, safeguards and three checkpoint boundaries.

Compare `separate` and `reuse` on original_free, matched_free and rounded.
The separate path copies the frozen training arithmetic; the reuse path
replaces only the joint loss computation with the qualified per-call helper.
Record actual shared work once in disjoint routes. Logical rollout counters
are not physical operation counts or backward FLOPs.

Run one warmup pair per arm, then three timed paired rounds, for 24 total
fits and 2,304 optimizer updates. Retain warmups and exclude them from timing
ratios only by this prior declaration. Round 0 uses canonical arm order.
Rounds 1, 2 and 3 rotate arm order by offsets 0, 1 and 2. A pair runs separate
first exactly when `(round + canonical_arm_index) % 2 == 0`. Three timed
pairs imply a disclosed 2:1 or 1:2 within-arm order imbalance. There is no
warm start, timing-dependent order, dropped outlier or replacement repeat.

The primary time encloses the complete `train()` call: construction, all
updates, state snapshots, optimizer state, checkpoints, allocation trace,
fit-row write and final guard. Directory creation and outer fit-record writes
are excluded consistently. Dataset generation, source verification, startup,
warmups, audit and total native-phase elapsed are reported separately, not
hidden within or added twice to the primary time. Preserve the original
internal fit and stage clocks as diagnostics, not alternative primary scores.

Before reading model arrays, the independent saved-output audit authenticates
the registered sources, original producer receipt, exact inventory and closed
native process. It checks the complete 24-fit roster, every accepted update,
paired indices and support reconstructed from saved inputs, exact initial and
prefix-boundary model/Adam states, and final model/Adam values with prospectively
fixed absolute and relative tolerance 1e-7, using the symmetric per-entry bound
`abs(a-b) <= 1e-7 + 1e-7 * max(abs(a), abs(b))`. Report actual discrepancies.
The audit must not call a model, optimizer or world generator. Test integration
against the frozen train path separately before admitting the benchmark.

Report each paired ratio `separate call seconds / reuse call seconds` and the
median ratio per arm. The engineering speed gate requires every timed ratio
strictly above 1 and every arm's median ratio strictly above 1.05, with all
numerical and provenance checks passing. This is a small, single-machine
throughput result, not statistical significance, an architectural advantage,
equal compute, or evidence of improved task performance. Failure leaves the
gate failed; it cannot trigger more repeats or looser numerical tolerances.

Freeze source hashes, source snapshots, runtime, schedule and commands once.
Run three separate native-supervised phases: qualification 180 seconds,
producer 120 seconds and saved-output audit 60 seconds. Producer additionally
has a 90-second cooperative whole-producer bound and each fit retains a
30-second exact-controller cap. Single numerical thread; ambient pytest
plugins and bytecode writes disabled. Sample producer peak RSS at checks
(4 GiB) and cap its retained output at 512 MiB/2,048 paths at checks; do not
claim continuous memory or output enforcement. Any failed/incomplete phase
stops this registration. Preserve partial artifacts and original closures.
Do not edit frozen sources or rerun this registration.
