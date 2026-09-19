# Frozen-expert coordination screen

This development experiment follows the completed support-weighting screen. Its
recency-plus-Huber predictor is strongest on three pooled physical endpoints;
the existing GRU is stronger on zigzag rotation. Both archives have already
informed design. They are not fresh confirmation data.

The question is whether a small recurrent selector learns useful coordination,
beyond choosing a fixed combination of these complementary predictions. A pass
would justify fresh confirmation, not establish an architectural contribution.

## Frozen experts and legal inputs

Use the three final meta and three final GRU checkpoints from pose-adaptation-v1.
The fast expert uses the fixed decay_huber3 support policy from pose-support-v1.
Each expert privately forecasts all25 steps from exactly32 observed poses and
31 completed action blocks, conditioned on25 supplied future recorded applied
torque blocks. Never feed a blended pose back into either expert.

The selector receives31 completed-transition tokens. Token t-1, t=1..31, contains
body-frame vertical, backward displacement and spatial rotation increment scaled
by the existing training-only scales, followed by completed action[t-1]. Apply
tanh to the49 concatenated coordinates. The latest observed pose31 is included.
No archive label, parent ID, raw start, future observation or future action is
available to the selector. Its two sigmoid outputs are fixed over the horizon.

Alpha_p and alpha_R are fast-expert fractions. Positions combine linearly.
Rotations use Exp(alpha_R Log(R_fast R_slow^T)) R_slow. Hard0/1 weights return
exact expert endpoints. A mixed forecast is a valid sequence of poses; it does
not establish that both outputs share a physically consistent latent state.

## Fixed controls and training

Evaluate eight configurations for each of3 paired seeds:

- fast alone; slow alone;
- fast position with slow rotation; the reverse composition;
- fixed half mixing;
- two learned constant logits;
- a non-recurrent selector: last token, mean token and last-minus-first token
  concatenated into147 coordinates, Linear147->22, tanh, Linear22->2;
- primary recurrent selector: GRU49->16 over31 tokens, Linear16->2.

Every selector's final layer starts at zero, giving exact initial half mixing.
The context selectors have similar parameter counts, but deliberately different
access to temporal detail. This comparison does not isolate recurrence from
information compression: the summary model sees three predetermined summaries.

Only selector weights train. Cache both unchanged expert forecasts on the
original720 training windows once perseed. The experts previously trained on
these same windows, so this is in-sample stacking, not out-of-fold stacking.
Fit all3 selector types on seeds1101,1202,1303,30epochs,batch32,Adam.001,
gradient norm cap1. Every paired arm receives identical epoch permutations.
Use final checkpoints after690updates each:9fits,6210new optimizer updates.
All25 physical forecast steps contribute the original position/.1m and
geodesic-angle/.1rad squared-error objective. No selection, sweeps, retries,
replacement seeds, panel-dependent rules or exclusions.

## Evidence and continuation

Bind source, protocol, data, previous sealed study/audit and expert checkpoints
before fitting. Save all initial/final selector weights, losses, batch orders,
training tokens/cached predictions, target IDs, full forecasts, per-case gates
and timing samples. An independent saved-output auditor reconstructs blends
with NumPy, all physical errors and the complete continuation gate. It does not
rerun predictors or optimizers. Training behavior remains source-bound.

Blend reconstruction tolerance: float32 rtol2e-6,atol2e-6, exact hard endpoints.
Active expert replay against the prior sealed run retains rtol1e-6,atol2e-7.
Independent cached training-token reconstruction uses rtol1e-4,atol1e-4
because small angular scales amplify float32 geometric cancellation.
All outputs must have proper rotations to1e-4 and finite values. An invalid
artifact cannot be counted as an empirical failure or success.

Retain the20 previous configurations and add five non-primary combinations as
controls. Replace inherited fast/GRU timings with freshly measured active expert
timings; deduplicate their identities in the gate. For recurrent versus each
of25 controls on both physical endpoints and both panels, require:

1. pooled RMSE at least10% lower;
2. all3 paired fit MSE values nonworse;
3. at least8/10 source parents nonworse;
4. strictly lower pooled MSE after every leave-one-parent-out removal.

Also require median full forecast cost at most1.5 times the freshly timed GRU.
This retains17 conjunction groups and1501 elementary comparisons. Earlier failed
studies remain failed. A narrower rotation improvement cannot rescue a failed
full gate. All160 windows perpanel, including source-start transients, remain.

Use CPU one thread, three warmups and20 timed individual windows for every row.
Charge context processing, both experts, gate and blending for combinations;
single-expert rows run only that expert. Loading, normalization, artifact I/O
and metric reporting are excluded. Training caches and gate training costs are
separate; previous expert training is inherited and disclosed. Gates do not
save expert computation. Timings are sequential local measurements, not a
hardware-independent speed claim.

## Prior art and claim limits

Recurrent expert gating is established, including
[Namikawa and Tani](https://arxiv.org/abs/0706.1317v2).
[Multi Time Scale World Models](https://papers.nips.cc/paper_files/paper/2023/file/54d8aab579b5a9ed3395764c7341ebec-Paper-Conference.pdf)
uses a coupled probabilistic hierarchy, which this screen does not reproduce.
[Test-Time Mixture of World Models](https://arxiv.org/abs/2601.22647)
adapts routing for embodied agents, another reason to avoid broad mixture novelty
claims. Here routing is learned offline and fixed at deployment.

No connectome, calibrated uncertainty, new RL, real-robot control or ICLR-ready
result is claimed. Forecasts condition on recorded applied torques, with
inferred Euler conventions. Raw/prepared data, training tokens and full
predictions remain local under unresolved upstream licensing. Publish our
implementation, numerical errors, receipts, weights and diagnostic summaries.
