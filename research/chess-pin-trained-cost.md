# Complete native cost of trained pin policies

The original cost-v1 protocol remains bound to interrupted quality-v2.
The separately frozen [cost-v2 recovery](../evidence/chess-pin-trained-cost-v2/README.md)
applies the identical comparison to quality-v3. It was frozen while zero
quality-v3 fits had completed and evaluation outputs were absent. Only the
version and explicit provenance bindings change; all scientific fields,
positions, methods, ordering and budgets remain equal. Forty-four constructed
fixture tests passed, including all fifteen original cost tests. No trained
timing has run and there is no new speed result.

The separate timing protocol is prepared while the matched quality study is
still training. It measures all nine policies: the frozen backbone, WLDN,
joint, separable, pairwise, root-only, counts, graph MLP and edit-labelled union.
No fitted policy or new quality outcome is read during preparation. Timing
requires the completed 24-fit quality study and its full replay audit.

The fixed panel is the first 128 ordinary development positions, in source
order, for all three backbone seeds. It contains 3,614 legal candidates,
926 pin-factor rows, 28 positions with root pins and 80 positions where at
least one candidate changes a pin witness. There are 280 such candidates.
These counts describe inputs, not prediction quality. No position is replaced
after inspecting coverage. The ordinary panel remains previously exposed
development data, not independent confirmation.

Each call starts from FEN and includes board and legal-candidate encoding,
one-position backbone computation, every required graph and pin input, the
policy head, argmax and score-list serialization. Model loading, numerical
comparison and file writing are outside the timer. Input work unused by a
policy is omitted: the backbone does not build graphs or pins, WLDN and the
union model do not build pins, and root-only extracts root witnesses once.
There is no persistent position/feature/graph cache between timed calls.

Nine rotating repeats put every method in every order slot for each seed and
root. All 31,104 timings are retained, along with 54 separately recorded
warmups. Report median complete milliseconds, per-seed medians and median
paired ratios to WLDN. Predeclared strata distinguish presence/absence of root
pins and candidate pin changes. These are scalar, shared-host measurements;
they do not establish throughput or hardware-general speed.

Every warmup and timed call retains its full legal score vector and selected
move. Compare both with the audit-authenticated full-native quality output,
using the unchanged 1e-5 score tolerance and exact argmax. Finite numerical
failures are recorded and fail the separate cost-path gate; structural and
nonfinite failures stop execution. Descriptive timing is allowed after a
completed failed quality study, but cannot change its scientific or numerical
gate. The original native/cached failures, if any, remain visible separately.

The cost audit authenticates the full saved run, recomputes every comparison
and timing statistic, and freshly repeats all 3,456 seed/root/method decisions.
It retains the complete fresh vectors. Recorded durations are authenticated,
not rerun. This shares neural kernels with the quality study and is not an
independent neural implementation. Primary execution and audit each have a
fixed 30-minute ceiling, including input authentication and setup.

Fifteen new tests cover all nine optimized paths against the complete native
path with nonzero heads, including a joint head after three actual fixture
updates. Positions cover pins, empty factor inputs, castling, promotion and
en passant. Other checks cover skipped unused work, independent pin detector
coverage, balanced timing order, paired arithmetic, retained failures, missing
or corrupt records, pre-outcome ordering, full-audit prerequisites, deadlines,
and a complete fixture execution/replay. Together with the existing quality
runner and training mechanics, 38 tests pass. These are engineering checks,
not measured chess quality or trained-policy speed results.

The [original protocol](../evidence/chess-pin-trained-cost-v1/README.md) and
[current recovery](../evidence/chess-pin-trained-cost-v2/README.md) remain separate
from the active quality study. The manuscript should incorporate
trained quality and cost only after their complete results and audits exist.
The overall paper still needs a supported novel contribution, independent
confirmation and gameplay evidence.
