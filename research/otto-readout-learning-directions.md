# Learning rules to test after the readout/state ablation

These are conditional research directions, not registered experiments or claims
of novelty. The [closed ablation](otto-readout-state-ablation-results.md) is mixed:
full joint passes only 3/6 paired cells, and the action-only readout is within 5%
of its gap in only 3/6. Neither branch below is established across all cells.
The reused development paths can guide the next proposal but cannot confirm a new method.

## If a small readout is sufficient

[Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)
treats a small model's weights as recurrent state and updates them through a
learning objective. A useful OpenJev variant would adapt a small score readout
only after a permitted teacher observation, with its initialization or update
rule learned on TRAIN episodes. Prediction must occur before each write.
Using teacher-score targets would make this our supervised adaptation variant;
it would not reproduce the paper's self-supervised reconstruction task.

Compare the identical starting backbone/readout against a fixed head, ordinary
online gradient descent and the existing RLS estimator. Match label access,
query schedule, data and fit seeds. Charge inner-loop updates and outer training.
A close offline readout result alone does not establish useful online adaptation.

## If updating recurrent weights is necessary

[e-prop](https://arxiv.org/abs/1901.09049) combines forward eligibility traces with
learning signals to approximate temporal credit assignment. The useful question
would be whether an explicit local recurrent update retains the benefit of joint
training at lower training cost. First derive and check the rule for the actual
cell; the published work is not a drop-in justification for an arbitrary GRU
update. Keep the architecture fixed while changing the learning rule.

Compare against matched backpropagation through time, with identical initial
weights, labels, loss and exposure. Remove or reset eligibility traces in a
separate control. Report utility against measured time and memory. An approximate
update on a conventional network is not evidence of biological plausibility.

## Existing delta memory is a control

[DeltaNet](https://arxiv.org/abs/2406.06484) uses an error-correcting associative
update and contributes an efficient parallel training algorithm. Its delta-rule
memory is close to the trace-delta mechanism already tested here. The existing
negative result stays visible. A new comparison would need an identified
interference problem, matched keys/values and memory size, additive versus
error-correcting writes, and paid gate/update computation. Renaming the existing
rule would not add a contribution.

If all continuations hurt the parent, investigate the training objective and
transfer behavior before adding another learning mechanism. A match by the
both-readout arm means recurrent **weight updates** may be unnecessary under
this recipe; its base readout still changes recurrent trajectories through
prediction-error feedback. None of these score-forecast results, on their own,
establishes a learned world model, a connectome advantage or autonomous utility.
