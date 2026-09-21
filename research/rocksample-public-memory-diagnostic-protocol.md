# RockSample public-memory opportunity diagnostic

21 September 2026. This diagnostic asks whether older public history improves
predictions of noisy check outcomes under a fixed exploration policy. It is not
an architecture contest, learned result, task-return study or benchmark score.
The unchanged native POBAX task is used after runtime qualification. The policy
avoids the eastern exit while collecting 256-step trajectory fragments. This is
a deliberate exploration distribution, not an estimate of a trained policy's
memory needs.

## Fixed collection and information contract

- POBAX commit `a5e1d62d14e4efe783885b9d4f19cffa2a568eec`, RockSample (11,11),
  Gymnax 0.0.9, JAX 0.6.2, CPU, resolved isolated runtime.
- Eight constructor map seeds: 11001 through 11008. Four reset seeds per map:
  21001 through 21004. Action and transition randomness derive from separate
  fixed generators, never from results. No model has been trained on these maps.
- 256 transitions per fragment, 32 fragments, 8,192 transitions. No hidden state
  injection. Reset through the normal public interface. Check actions are
  uniformly selected among 11 rocks with probability 0.2, sample with probability
  0.2, and each movement direction with probability 0.15. An east move from public
  column 9 becomes west to keep the fragment within the task. No agent observes
  the constructor/reset seed, true map, qualities, or reward.
- Allowed model inputs: signed 33-dimensional observation, previous action,
  episode boundary, and published task constants (board size, rock count and
  sensor rule). Sampling updates follow the known task rule at the public
  coordinate. The table ignores without-replacement coupling between rocks.
- Record raw public transition traces. Store map/reset identities only as audit
  metadata. Do not record true qualities or coordinates in actor data.

Before any native diagnostic run, analytical support review reduced the check
rate from the draft 0.5 to 0.2. At 0.5, a queried rock's chance of a 128-step gap
would be about 0.0026, making the required delayed subset implausibly small.
At 0.2, the corresponding probability is about 0.095; accounting for finite
fragments predicts roughly 47 delayed check endpoints. The minimum remains 32,
and actual support can fail. This is prospective collection design, not a change
made after observing predictor scores.

## Predictors and measurement

Predict each selected check's signed outcome **before** executing/incorporating
that check. Evaluate all predictors on identical traces and endpoints:

1. Prior probability 0.5.
2. Full-history factorized location/quality filter, 2,420 probabilities.
3. Same filter reconstructed from the last 32 completed transitions.
4. Same filter reconstructed from the last 128 completed transitions.
5. Same filter reconstructed from the most recent check of the queried rock
   and all public actions since it; if none exists, use a fresh prior followed
   by all prior actions. This is a latest-check control, with subsequent sample
   transitions, not an optimal sufficient statistic.

Report check NLL and Brier score overall and for queries whose most recent check
of that rock occurred more than 128 transitions earlier. Average within each
constructor map then across maps with nonempty support; identify the number of
contributing maps for the delayed subset. Publish every map, all endpoint counts and
paired differences. Exact zero/one impossible-event probabilities produce infinite
NLL and failure, rather than numerical clipping that hides a defect. No tuning,
replacement seeds or conditional extra collection. Model differences arise only
from retained public history. Replay cost is not a deployment speed benchmark.

## Continuation rule and bounds

Only admit a subsequent learned memory pilot if full history improves mean NLL
by at least 10% against **each** recent-128 and latest-check control, with positive
paired gain on at least six of eight maps for each, and the delayed subset has
at least 32 check endpoints spanning at least four maps. This is an exploratory
opportunity gate, not a significance claim. Failure closes this collection as a
justification for a larger memory model; it does not disprove recurrence generally.
A later policy comparison must establish raw-return gains under matched planning
and compute before claiming useful control or a new architecture advantage.

Runtime cap: 600 suspend-inclusive seconds, 8 GiB process peak RSS, 128 MiB output.
One fixed run after synthetic qualification. Preserve partial outputs and failures.
