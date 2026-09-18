# Matched pin-factor quality study

**Status: v3 completed and passed its full replay audit; the scientific
continuation criterion failed, with 2/16 quality checks passing.** The joint
model trailed every trained comparator in mean agreement on both panels.
[All families and interpretation](../docs/chess-pin-quality.md).

V2 disappeared before evaluation after five completed fits and 8,239
logged updates. Its artifacts and costs are preserved in an
[interruption review](../evidence/chess-pin-quality-v2/interruption-review.json).
The cause is unknown. A separately frozen
[v3 recovery](../evidence/chess-pin-quality-v3/README.md) repeats the same
scientific protocol from fresh initializations with a detached launcher.
V1's earlier schema stop remains preserved. Static notes do not establish
process liveness.

The [protocol](../evidence/chess-pin-quality-v2/protocol/plan.json) has SHA-256
`9f9eba11e6abaa9b15c085a0098edbe938e7db848a323a3646f7f9468aadfc48`. It binds the executable runner, statistical analysis, tests,
canonical inputs, runtime versions, prior engineering audits, measured update
profile and the applied pre-outcome comparator rule. The runner is
`scripts/chess_pin_quality_study_v2.py`.

## Preserved v1 stop and narrow repair

The [v1 failure review](../evidence/chess-pin-quality-v1/failure-review.json)
retains all 770 logged completed WLDN-97 updates, the initial state, original
plan/source hashes and the 320.788-second cost. No fit completed and no
quality predictions were produced. The process was deliberately interrupted
because the frozen string-only bootstrap validator would reject the corpus's
integer source-game IDs after training.

V2 preserves homogeneous integer or nonempty string identifiers exactly,
including integer zero. It rejects booleans, floats and mixed types rather than
coercing identities. Before fitting, it checks the authenticated historical
metadata for all 4,096 panel records: 39 ordinary games and 31 shifted games.
All architecture, data, training, budget, numerical and quality criteria must
match v1, enforced by the new signature check. Every final head starts fresh;
no partial weights are reused. The interrupted attempt is retained separately
and is not evidence for or against the architectural hypothesis.

## Scientific question

Does joint processing of candidate-dependent absolute-pin witnesses improve
a small graph-difference chess policy beyond simpler representations of the
same motif and the strongest selected historical graph comparator?

The designated candidate is `joint`. It pools learned representations of the
king, pinned blocker and sliding attacker, with retained/added/removed pin
status. Its conditional set encoder is established methodology; the experiment
can test this specific inductive bias but cannot establish a new generic
architecture merely by improving a chess metric.

| Arm | Role in the comparison | Stored head parameters |
| --- | --- | ---: |
| WLDN | Established nodewise graph-difference baseline | 16,638 |
| Joint | Candidate with joint three-role factor MLP | 16,658 |
| Separable | Constant and separate role terms within the factor stage | 16,658 |
| Pairwise | Retains pair terms, removes anchored third-order remainder within the factor stage | 16,658 |
| Root-only | Root pin context without candidate-dependent factor changes | 16,658 |
| Counts | Six owner/status counts without witness-square identities | 16,656 |
| Graph MLP | Conventional added capacity without pin witnesses | 16,658 |
| Union edits | Strongest historical candidate selected by the pre-outcome rule | 16,740 |

All graph/action paths retain their declared information. These controls do
not remove every multi-piece interaction from the complete policy. Equal
stored parameters do not equalize active capacity, computation or optimization.

## Frozen execution

There are 24 fresh fits: eight arms and backbones 97, 109 and 127. Each uses the
same 32,768 original training roots, six epochs and 128-root batches, for 1,536
updates per fit and 36,864 total. The head seed, Adam recipe and batch ordering
come from the previously frozen shared mechanics. Initial and final states and
every minibatch index/loss/gradient-norm receipt are retained. No intermediate
checkpoint is chosen by evaluation performance.

All fits must finish before evaluation. Every declared head and the frozen
backbone are then freshly evaluated on both complete 2,048-root old panels:
110,592 position records, with all legal score vectors retained. Independently
of the stored feature/graph caches, reconstruct every native root/backbone
input and score every method. There are 12,288 native root/backbone input
reconstructions and 3,226,149 native/cached candidate-score comparisons.
Inputs may be reused across heads for one root; this is equivalence checking,
not complete-decision timing.

Every score difference must be at most 1e-5 and every chosen move must match.
All finite failures are retained and fail the numerical gate. Structural,
nonfinite or source-identity errors stop execution with a failure receipt.
The separate audit replays every cached score/NLL and native vector exactly,
checks initial states, final checkpoint identities, all training-index receipts
and training/evaluation ordering, and recomputes metrics and all criteria.
It shares production neural kernels and does not retrain the models; earlier
independent input/packing audits retain their separate scope.

## Acceptance and uncertainty

Require all sixteen panel/comparator checks: joint must gain at least one
mean agreement point over each trained comparator on each panel, remain at
least as good as the frozen backbone, and have no paired-seed deficit worse
than half a point in any comparison. Both the numerical and quality gates,
and the full evidence audit, are required before advancing this candidate.
A stronger control cannot be retrospectively renamed the treatment.

Report NLL descriptively. Retain 2,000 paired source-game bootstrap draws per
comparison, seed 996101 plus the shift indicator, after averaging the three
paired seed outcomes within each root. Position-weight the sampled games.
These intervals are conditional on repeatedly exposed development panels;
they do not supply new-seed uncertainty or correct adaptive selection.

The selected-arm profile projects 5.715 hours of updates.
Execution has a fixed twelve-hour ceiling, including setup and full evaluation;
audit has a separate two-hour ceiling. No timeout extension, resumed partial
fit, seed replacement, new engine/model call or copied historical prediction
is permitted. The separately predeclared
[native-cost comparison](chess-pin-trained-cost.md) now has a separately
frozen cost-v2 provenance update for v3 checkpoints, prepared before any
quality evaluation. It runs after the study and its audit complete and uses 128 fixed ordinary
positions, nine rotating repeats and complete native input construction.

## Validation before launch

Thirty-one tests pass across the new runner/analysis, shared training mechanics
and comparator selector. The nine runner tests cover source/target/menu
alignment, actual fixture training and checkpoint identity, complete minibatch
receipts, the all-fits-before-evaluation barrier, complete cached/native score
capture and exact replay across all selected arms, deliberate score corruption,
retention of finite numerical failures, hard deadlines and exact file membership. They also check the historical integer-ID schema
before training. The analysis tests include integer zero, mixed-type rejection
and Boolean aliasing. These use constructed fixtures and do not establish chess quality.

The [v2 evidence directory](../evidence/chess-pin-quality-v2/README.md) retains
its launch, tests and interruption review. V3's separate
[evidence directory](../evidence/chess-pin-quality-v3/README.md) binds the fresh
run at `runs/chess-pin-quality-v3/execution` and its detached supervisor.
Do not rerun because an observation call times out. The supervisor invokes
the frozen audit after authoritative primary completion.
The completed v3 result rejects advancement of the designated joint candidate.
Independent confirmation, actual gameplay and a defensible novel contribution
remain open requirements for the overall paper goal.
