# Four-gate learning pilot

**Development screen, specified before fitting.** This will test whether the
implemented memory gates learn useful public-observation predictions. It does
not measure native control, calibrated uncertainty, biological superiority or
untouched test performance. No results are present in this document.

## Data and fixed comparison

Use the existing 768-episode Reacher training corpus, authenticated by its NPZ
and metadata hashes. Read only public packets, issued commands and total
executed rewards. Hidden angles, velocities, actuator disturbances and native
state are not inputs or targets. Some recorded actions came from a privileged
IK/PD collector; these are off-policy recorded trajectories.

Select 128 whole development episodes before fitting: 32 from each of four
sensing phases, with fixed collection-policy quotas and a hash ranking within
strata. The remaining 640 whole episodes train every arm. This corpus was used
to train previous models, so the split is development data. No previously
fitted model that saw all 768 episodes enters this comparison.

The original public sequences have two six-packet sensing gaps. Development
has two panels: the original packets, and a derived panel that hides the next
four previously public packets after each gap. Recompute observation age after
censoring; retain commands and rewards exactly. Never fill originally missing
angles from audit state. This is a masking stress test on the same trajectories,
not a fresh native scenario shift.

Train constant, age, raw-error and normalized-error gates with three paired
new initializations. All arms use fast width64, context width16, eta0.1,
dt0.02, noise0.05, residual-moment bounds0.0001 to4, the same complete-episode
minibatch orders, and all the same instantiated modules. Eight epochs at
batch32 give160 updates per fit and1,920 updates across12 fits. No early stop,
checkpoint selection, automatic retry or in-run budget extension is allowed.

The objective is the existing one-step/open-loop mean and reward MSE, with
H5 rollout weight0.5 and reward scale4, plus residual-moment score weight0.1.
Adam uses lr0.001, betas0.9/0.999, epsilon1e-8, and no weight decay, fused or
foreach path. Clip the variance-head and other parameters separately to norm10
before the single Adam step. A global clip would let auxiliary-head gradients
rescale the backbone despite the detached objective; separate clipping avoids
that indirect coupling. Record both preclip norms.

All12 fits must finish before development evaluation. Evaluate initialization
and final weights only, retaining every fit and both sensing panels. The source
revision, public split, actual initial tensors, full epoch orders, complete
configuration, runtime and finite wall-clock cap must be frozen before launch.
The outer process has a fixed1,800-second cap including launch checks and
terminal writes. No numerical runtime estimate or execution status is implied
by this design. Capacity attempt01 failed on non-plain runtime metadata after
three synthetic updates. Attempt02 corrected the version string, completed
three B32/T50 updates per gate and eight B32 development batches, and preserved
both attempts. These synthetic checks establish runnable shapes and rough
capacity only. They do not establish model quality or comparative speed.

## Metrics and continuation

The primary metric starts **after** assimilating each returning observation:
predict one and three steps ahead without any further observation. Average MSE
equally over four cosine/sine coordinates, horizons1 and3, the two recovery
points, then episodes. Missing coverage is a failure to provide the metric,
not permission to select an easier subset. Error of the prior at reacquisition
is a separate diagnostic, since that packet's correction cannot change its
own prior. Also retain ordinary one-step angle/reward MSE, full H5 predictions,
residual moment scores, variance-bound fractions and actual training cost.

Continue this mechanism to a separately specified confirmation only if all
fits/evaluations complete with finite evidence and all of these rules pass:

- In each sensing panel, normalized gating's final family-mean recovery MSE
  is at most97% of both age and raw-error gating's family means.
- Every paired normalized fit is nonworse than age and raw-error gating in
  each panel; normalized family mean is also nonworse than constant gating.
- Against age and raw-error gating in each panel, normalized family mean has
  at most2% worse observed-target one-step angle MSE and5% worse reward MSE.
- Each normalized fit reduces ordinary-panel recovery error by at least10%
  against its own initialization.

These are practical development filters, not significance tests or independent
test-set claims. Report all comparisons, including failures. Weak learning is
inconclusive; a failed superiority screen does not prove equivalence and does
not authorize extending this run. The broader research goal remains open.

An age-conditioned empirical error-scale lookup, fitted only from training
residuals, is required in the next confirmation before attributing improvement
to learned uncertainty or moving to native-control comparisons. Conventional
filtering and matched recurrence remain necessary for a broader architecture
claim. Successful prediction alone is insufficient evidence of better control.
