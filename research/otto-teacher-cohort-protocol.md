# Teacher-cost learning: fixed cohort eligibility

22 September 2026. Prospective, before decoding the larger cohort. This first phase determines whether a representative slice of the existing learner-TRAIN pool supports teacher continuations. It generates no labels and trains no model. The six-anchor pilot remains unchanged.

## Fixed selection

Authenticate the complete source and input closure of `output/otto-teacher-label-pilot-v1/plan-01.json` (SHA-256 `6956fd2658fc8920c277a6a4585ecb75d0b1639730ead5158af0cc7835269380`). Use only its original symmetry-study learner-TRAIN metadata, public transition journal and lambda3/lambda4 observation kernels. No validation or evaluation values, saved hidden sources, checkpoints or analytic costs enter this phase.

Require all 4,596 retained rows and all 144 declared episode identities: lambda3 seeds 950001-950012 and lambda4 seeds 960001-960012, each from `shared` and `dense` collectors at 9101, 9102 and 9103. Within each episode sort by `(prefix_index,row_index)`, including prefix zero. With `n` retained rows choose `k=min(4,n)` and positions `floor(j*(n-1)/(k-1))` for `j=0..k-1`; if k is one choose position zero. Assign anchor IDs after sorting by `(regime,seed,arm,prefix_index,row_index)`.

This rule gives at most 576 anchors and preserves initial and late retained states. Short episodes contribute every retained row and remain explicit. Missing episodes or duplicate identities are integrity failures. Selection never uses belief mass, eventual success, action gaps or teacher costs. Freeze exact selections before reconstructing beliefs. Historical retained-prefix sampling depended on trajectory length, so this is coverage of an existing training pool, not a random population sample.

## Public reconstruction and support

Use the separately reviewed generic cohort extractor, without changing frozen pilot code. Allowlist exact episode IDs on raw journal headers before numerical decoding. Project public observations, chosen actions, identities and posterior witnesses only. Replay each selected episode once through its last selected prefix, checking every original float64 posterior hash and mass and every action/position transition. Capture immutable exact beliefs at every selected prefix; prefix zero is captured after its reset hit, without assimilating that hit again.

After all captures, attempt the unchanged `TeacherSnapshot` constructor once for every selected anchor. Record all expected belief-domain rejections and continue assessing the remaining anchors. Do not turn a rejected constructor into a successful constructor count. Provenance, shared-kernel, resource and unexpected runtime errors fail execution. Never normalize, clip, replace or silently omit a state.

The phase is technically complete only if its original supervisor succeeds and every planned reconstruction and support assessment is accounted for. **All selected anchors must be supported before this cohort can proceed to continuation sampling.** Any rejected anchor prevents sampling and fitting this cohort; publish its identity and reason. A later changed method requires a distinct prospective design, not an in-place retry.

## Resources and evidence

One original process, 600 suspend-inclusive seconds, one numerical thread, 4 GiB peak RSS and 1 GiB output. At most 144 resets, 314,928 updates and 576 support assessments yield at most 631,296 attempt/return events. Enforce 1,024 serialized bytes per event and reserve 64 MiB for arrays, metadata and receipts. The bound is 713,555,968 bytes, below 1 GiB. Journal each attempt and return durably. Preserve partial outputs and failure receipt; use an exclusive directory and the existing suspend-inclusive supervisor. No retries, replacement states, source draws, native simulator, analytic choices, learned-model calls or optimizer calls occur.

Freeze source, test, protocol, interpreter, installed-package and historical input hashes. Run the new extractor's fabricated tests before empirical preparation, retaining their original exit status. The final report must join the original worker and supervisor and independently check selected IDs, copied posterior hashes and full assessment counts. Report complete elapsed time and output size, not just filter time.

## Intended learning comparison after eligibility

If all anchors qualify, freeze a separate sampling, fitting and fresh-evaluation allocation before generating any labels. Compare the same ordinary 2,836→32→16→4 head under two target sources, using common eligible-action centered regression, identical D4 augmentation, paired initializations, episode weights and optimizer work. One arm represents the historical analytic preference logits; the other uses continuous paired teacher-continuation costs with one TRAIN-only global scale. Do not amplify near-ties by normalizing each continuation panel's range or selecting hard winners.

This would test supervision under a new common loss, not reproduce the old cross-entropy study or establish a novel architecture. Autonomous full-horizon competence, complete compute, scenario shift and paired controls remain necessary. The exact public posterior already summarizes history; recurrence cannot be credited with recovering omitted memory in this interface.
