# Next: integrate fixed rounding into the recurrent learner

The [rounded transition primitive](finite-rounded-transition-qualification-results.md)
passes its first 36 numerical checks. The earlier [fixed64 integration
attempt](finite-balanced-learning-stop-results.md) remains failed and closed.
No new integration or scientific learning study is admitted by this note.

Use a new model adapter and new registrations. Preserve the closed fixed64
sources and artifacts. The rounded transition and its final log probabilities
must be shared by prefix likelihood, prefix-only prior, filtering and
forecasting. Keep the emission and hazard fields, public-history interface,
H1/H2 supervision, cost readout and prefix-then-joint schedule unchanged.
Balance T only; retain state-dependent survival losses in `diag(1-h) T`.

Compare original column-softmax, a free control initialized from the new
rounded T, and the persistently rounded model. Verify their initial effective
functions independently. Retain all initial, boundary and final model/Adam
states, rejected updates, full timing and the actual normalization/correction
work. Report the correction magnitude so improvements cannot be attributed
only to enforcing row sums when smoothing or gradient geometry also changed.

First run one separately frozen engineering integration on non-scientific
histories, including an independent saved-output audit. A passing primitive
is not sufficient, as the previous stopped integration demonstrated.
After that passes, register fresh scientific histories and seeds, with all
nine final checkpoints before development generation. Keep the same
10/40-second allocation and the original all-seed SHORT/BLIND/OBSERVED gates,
paired H4/H8 regret checks, minimum ten-percent mean improvements against each
control, and complete-fit-time bound. Do not reuse the earlier stopped
registration or inspect development outcomes while choosing the function.

A successful comparison would still need untouched replication and a world
whose true transitions are not doubly stochastic. Known stochastic structure
and privileged starting readouts limit generalization. This step is about
making the learning comparison executable; it is not evidence for a novel
connectome architecture, memory preservation or robotics transfer.
