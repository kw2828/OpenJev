# Better readout fitting does not resolve the recurrent failures

**All nine linear decision heads meet the numerical accuracy requirement, but
every model family still fails the short-horizon and blind-forecast criteria.**
This completes the diagnostic that the earlier solver could not finish. It
does not advance a reliable policy or establish a new architecture.

![All nine frozen recurrent models, before and after fitting their decision heads](finite-gap-readout-study-results/benchmark.png)

[Complete tables and paired differences](finite-gap-readout-study-results/report.md) ·
[Frozen protocol](finite-gap-readout-study-protocol.md) ·
[All models, outputs and receipts](https://github.com/kw2828/OpenJev/releases/tag/finite-gap-readout-study-v1)

## What changed

The nine existing factorized, initially matched unrestricted, and dense
unrestricted recurrent checkpoints stay frozen. Only the bounded linear cost
head is refitted on their original H1/H2 training states. The solver uses a
qualified projected-gradient method and the unchanged 1e-8 optimality-gap
threshold. Each original head is evaluated beside its solved head on the same
fresh cases. No H4/H8 training targets or additional recurrent updates are used.

All nine solves certify in 10-250 iterations, within the fixed 20,000-iteration
cap. Each training objective falls. The independent audit reconstructs final
objectives, gradients and certificates from saved states. Event, survival and
state predictions remain byte-identical between paired views.

## Decision performance

Mean regret across three seeds, lower is better:

| Frozen model | H4 original | H4 refitted | H8 original | H8 refitted |
| --- | ---: | ---: | ---: | ---: |
| Factorized | 0.224308 | 0.221728 | 0.228146 | 0.228278 |
| Matched unrestricted | 0.284370 | 0.272996 | 0.314550 | 0.301921 |
| Dense unrestricted | 0.326170 | 0.333721 | 0.382055 | 0.386487 |

These are separate-policy averages, not an ensemble or a significance test.
The modest improvement for the matched model does not satisfy the original
all-seed criteria:

| Model, original then refitted | Short horizon | Blind forecasts | Observed filtering |
| --- | --- | --- | --- |
| Factorized | FAIL 23/24, 23/24 | FAIL 15/21, 15/21 | FAIL 6/8, 6/8 |
| Matched unrestricted | FAIL 23/24, 23/24 | FAIL 13/21, 13/21 | PASS 8/8, 8/8 |
| Dense unrestricted | FAIL 22/24, 22/24 | FAIL 11/21, 10/21 | FAIL 4/8, 4/8 |

DEV contains 128 attempted histories, including 14 terminal histories and 114
eligible forecast cases. All attempts are retained. This is a new development
sample after inspecting prior studies, not untouched final confirmation.

## Qualification, failures and evidence

The [separate numerical prerequisite](finite-gap-solver-qualification-results.md)
passed 18/18 synthetic fixtures versus 10/18 for the original solver. It made
no empirical model calls. That result qualified this new registered study; it
did not reopen the [stopped predecessor](finite-convex-readout-results.md).

The first integration attempt failed one of 75 tests: a tiny nonzero reported
export repair was incorrectly accepted by a tolerant audit comparison. Its
complete source snapshot and failure are preserved. A separately versioned
auditor requires exactly zero repair; the second integration passes all **75
selected tests** and lint. The scientific fit and independent audit each
complete once. Their original elapsed times are 2.270 and 1.860 seconds;
successful integration takes 2.318 seconds. These are local diagnostic timings,
not an inference-speed comparison.

The initial report renderer also stopped because it omitted the source
snapshot's manifest from its expected file roster. That publication failure
and both original helper files are archived. The corrected presentation tools
include the manifest explicitly; they change no scientific output or source.

The release includes all nine parent checkpoints, original and solved heads,
state caches, data attempts, predictions, numerical histories, independent
checks, source pins, and failed engineering/publication attempts. The historical
failing audit test is retained in the evidence archive rather than installed
as a current test. The passing audit version is committed. This is not a claim
that every unrelated test in the repository was run.

## What this rules out, and what it does not

Within this constrained linear family, the training head is now numerically
near-optimal. Refitting it does not resolve the observed decision failures.
The certificate concerns the training squared-loss objective, not generalization
or decision regret. It does not prove that the latent states lack useful
information, that nonlinear heads cannot help, or that the world is unlearnable.

The next proposed intervention changes [how hidden dynamics are learned](finite-gap-readout-study-next.md).
The earlier failed replication and its longer-horizon training follow-up stay
closed. Scenario transfer, a second environment and evidence of a distinct
mechanism would still be required for a broader paper claim.

Study registration SHA256:
`6b0a6d9330a6add78c6e91357eb19a30e78a960c953f7c4da305f8d1160589b8`.
