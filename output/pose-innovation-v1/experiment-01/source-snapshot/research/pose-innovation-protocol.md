# Does ordered error history predict useful corrections?

This is a fixed training-parent mechanism screen. It is a prerequisite for a
possible recurrent updater, not a new world-model result. Previous residual
RLS, learned ridge adaptation and past-error blending failed their continuation
rules. Merely adding signed residual corrections would repeat those mechanisms.

The narrower hypothesis is that the **order of completed prediction errors**
adds useful information beyond their summaries, ordinary observation history,
and linear models of ordered history. We test a direct forecast-error head
before investing in a correction mechanism inside recurrent dynamics.

## Data and exclusion

Use only the prepared original training archive, 30 parents with 24 overlapping
windows per parent. Each window contains 32 observed poses and 25 forecast
poses at nominal 20 ms intervals. Each transition has 40 ordered recorded
torques. They are applied torques, not authenticated planned commands.

Reuse the nine frozen body-GRU experts from `pose-crossfit-v1`: three seeds
(1101, 1202, 1303) and three folds. Expert f was trained on the 20 parents whose
IDs modulo three differ from f. Its action and motion statistics come from
those same parents. Both correction-training and correction-test parents are
excluded from that expert's training and fold statistics.

Within the remaining ten parents, let rank = parent ID integer-divided by
three. Ranks 1, 4, 7 and 9 are correction-test parents; ranks 0, 2, 3, 5, 6 and
8 are correction-training parents. Every fold uses 144 training and 96 test
windows. Across folds there are 12 distinct test parents and 18 correction
training parents. All three seeds use identical parent assignments.

Windows overlap within a parent. Parents are the aggregation units; windows
and seeds are not independent replicates. Each test parent is excluded from
its own prediction pipeline, not from every backbone in the collection.
Earlier research used the original training data; this is development evidence,
not newly collected or untouched confirmation data. The dev, plain and zigzag
panels will not be loaded or evaluated in this experiment.

## Predictions and features

For each of the 31 completed transitions, advance the frozen backbone before
assimilating the actual successor. Save that one-step prediction, then
assimilate. The deployment forecast retains the final actual motion (CV1) and
advances for 25 steps. Require parity with the authenticated earlier fold
forecast. Do not use the generic model forecast method, which installs CV16.

Each context token contains the established 49 public features and six signed
one-step errors. Public features are current-body vertical, measured backward
translation and angular motion, and the recorded action. Errors are expressed
in the final observed root31 frame and divided by frozen fold motion-change
scales; tanh follows concatenation. This is a retrospective encoding of known
history, not a claim that the root-frame tokens existed at earlier times.

Heads emit 25 six-dimensional corrections. Translation is added to the base
forecast; rotation is left-composed using the exponential map. Both are
expressed at root31 with output scales 0.1 m and 0.1 rad. Corrected poses do
not feed back into the frozen backbone. The correction head has no future
action input; the base forecast does. This limits the diagnostic to a
history-conditioned correction of an action-conditioned forecast.

## Nine fixed methods

1. Frozen backbone with zero correction.
2. Shrinkage bias: sum the 31 completed one-step residuals, divide by 32 and
   multiply by forecast lead 1 through 25. The zero-prior shrinkage is fixed.
3. Summary ridge using last token, mean token and last-minus-first token.
4. Ordered ridge using the flattened full token sequence.
5. Summary MLP: 165 inputs, nine tanh units, 150 outputs; 2,994 parameters.
6. Ordered GRU, primary: 55 inputs, eight recurrent units, 150 outputs;
   2,910 parameters.
7. Shuffled GRU with the same architecture and initial weights. Permute whole
   interior tokens independently per case, preserving first and last.
8. No-error GRU, same architecture and initial weights, zeroing only the six
   residual channels.
9. Error-shuffled GRU: preserve the 49 public features in chronological order;
   permute only the six interior error channels using the same per-case
   permutation. First and last errors remain unchanged.

Shuffling uses a fixed per-case permutation, independent of labels, from
NumPy seed `seed + 10000 + fold`. Summary features are mathematically invariant
to those interior permutations. The fixed small MLP has 2.9% more parameters
than the GRU; exact parameter or compute equality is not claimed.

Whole-token shuffling alone also removes public-history order. The ninth arm
tests the temporal organization and alignment of explicit errors conditional
on the same ordered public features. A difference does not separate error
sequence order from its alignment with the corresponding public states/actions.
This control was added during source review, before freezing or real-data calls.

For ridge, L2-normalize each feature row, center on correction-training rows
only, and solve dual ridge with fixed precision one and an unpenalized
intercept. Fit all 150 tangent-residual outputs together. This optimizes a
tangent squared-error surrogate; neural models optimize physical pose error.
The ordered ridge has more coefficients than the neural models and is an
intentional strong linear control, not a parameter-matched model.

## Fixed training and stopping rule

Initialize all neural output heads to zero. Use the same seed and minibatch
orders for paired neural methods. Train 40 epochs, batch size 32, Adam at
0.001, gradient norm cap one: 200 updates per fit, 45 fits and 9,000 updates.
Use only final checkpoints. No tuning, retries, selection or dropped cases.
The backbone has no gradients or updates. Optimize position squared distance
divided by 0.1 squared plus rotation geodesic angle squared divided by 0.1
squared, averaged over all 25 leads.

Advance this recipe only if the primary satisfies **all** of the following
against every one of the eight controls, on both physical endpoints:

- Pooled RMSE is at least 10% lower.
- Each of nine paired fold/seed MSEs is nonworse.
- At least ten of twelve test parents have nonworse MSE, pooling seeds.
- Each of twelve leave-one-parent-out pooled MSEs is strictly lower.

These are 368 overlapping descriptive comparisons, not independent
significance tests. They are a new training-only signal prerequisite, not a
replacement for the previous exposed-panel continuation rule. Failure closes
this fixed recipe without reopening the plain/zigzag panels. A pass would
justify testing an actual dynamics updater and then fresh generalization;
it would not establish either result by itself.

Save all predictions, correction vectors, initial/final checkpoints, losses,
orders, input identities and source snapshots. Independently reconstruct
geometry, controls and metrics from saved outputs. No audit neural reruns.
Report all training/test errors and paired/parent comparisons. Cached-head
latency is descriptive: three warmups and twelve single-window timings per
neural fit, including correction application but excluding backbone, cache
construction, permutation and I/O. It is not deployment latency.

Use one CPU thread and deterministic Torch algorithms. Freeze source, data,
parent checkpoint and normalization hashes before the first actual call;
retain a failure record if execution fails. Public outputs omit raw third-party
data and full forecast caches under the unresolved upstream data license.
