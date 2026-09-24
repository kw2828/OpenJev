# Qualify stable conditional dynamics before adding graph memory

The [robot coupling pilot](robot-coupling-results.md) closes with a failed
continuation rule. Rewiring performs slightly better than the physical chain,
GRU has lower mean error and latency, and every polynomial training recipe is
unstable. Do not promote a favorable seed, remove the failed comparator from
the rule, or open confirmation under this result.

The useful next question is whether the transition model is adequate before
assigning useful work to long-lived edge memory. A defensible next step is to
qualify a stable conventional recurrent dynamics model, then ask whether a
smaller learned transition can preserve its quality and reduce total cost.

1. Establish a competent stable baseline on FIT and already-exposed DEV.
   [ReLiNet](https://www.ijcai.org/proceedings/2023/0385.pdf) uses an initializer
   and state-dependent linear transitions; its stable variant constrains those
   transitions. The paper also reports a strong initialized-LSTM control.
   Borrowing either is baseline work, not our proposed novelty. Public
   [supplement code](https://github.com/AlexandraBaier/Supplement_ReLiNet) needs
   a compatible pinned dependency and equation checks before using its scores.
2. Separate two tasks prospectively. A causal dynamics prediction at step k
   may use input torques only through the transition into step k. An offline
   conditional forecast may use the entire realized future torque sequence.
   The current large direct ridge is the latter. A future causal direct-history
   control should mask later inputs per target horizon, with training and
   scoring rules fixed before the run.
3. Only after baseline competence, test one restricted transition mechanism
   against dense, low-rank and rewired controls at matched training exposure.
   Compare forecast error with actual latency and complete persistent storage.
   A smaller parameter count cannot substitute for a measured advantage, as
   the current graph-versus-GRU result demonstrates.

Both current DEV recordings are now exposed and must remain labeled development
data in any later search. The two reserved internal confirmation recordings and
official TEST remain unopened. A new proposed mechanism needs a new registration
and its own advancement rule; the old result does not become a pass. No new
training, confirmation access, control policy or RL experiment is registered by
this note.
