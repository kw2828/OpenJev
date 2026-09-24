# Full Bayesian covariance did not improve the recurrent predictor

**DEV FAIL: 4/13 conditions passed.** Evaluation and the independent audit both
completed. Full-covariance residual memory was worse than ordinary joint training
on the primary later-step decision gap in both settings. Its selected prior/noise
ratio was 0.1.

[Every method, fit seed and condition](otto-residual-reanalysis-results/README.md) ·
[Protocol](otto-residual-reanalysis-protocol.md) ·
[Registration](../output/otto-residual-reanalysis-v1/registration-01.json) ·
[Complete evidence archive](https://github.com/kw2828/OpenJev/releases/tag/otto-residual-reanalysis-v1)

![Nine methods and all three fit seeds, with full and later score gaps](otto-residual-reanalysis-results/methods.png)

## Primary result

| Sensing setting | Full RLS later gap | Best ordinary control | Candidate change |
| --- | ---: | ---: | ---: |
| Length 3 | 0.080167 | 0.061615, joint auxiliary training | **30.11% higher** |
| Length 4 | 0.115753 | 0.112005, joint auxiliary training | **3.35% higher** |

Lower is better. These are teacher-score gaps on recorded OTTO odor-search
paths, with equal originating-case and fit-seed weighting. They are not realized
search return, calibrated probabilities or autonomous gameplay results.

All six paired-fit later-gap nonregression checks failed. The four passing
conditions were technical completion, support in both settings and full-gap
nonregression in length 4. The rule required every condition, including at least
a 10% later-gap reduction in both settings.

Full versus diagonal covariance differed by less than 0.07% in later mean gap
at the selected ratio. The separate covariance contrast passed only **2/10**
conditions. Richer covariance was not the missing ingredient in this comparison.
The result does not rule out all recurrent memory or biological models.

## What was compared

All nine methods used the same 18 development paths, 4,816 retained states and
three original fit seeds. The frozen pretrained recurrent representation and
its trained trace projection supplied the residual-memory cues. Ordinary joint
auxiliary training was a separate stronger control. There were no new gradient
updates and no additional teacher or simulator calls during evaluation.

The controls were pretrained GRU, joint auxiliary GRU, last-error correction,
trace-delta correction, diagonal RLS and full RLS with correction multipliers
0.25, 0.5 and 0.75. Full RLS remained the sole candidate. All four prior/noise
ratios and all 72 views were retained. No control or best fit was promoted.

This is a separately registered reanalysis of paths collected by the
[earlier technically failed screen](otto-residual-runtime-repair.md). Registration
followed collection but preceded numerical evaluation. The original attempt
stays closed; these paths are prior development data, not held-out evidence.

## Completion and cost

| Phase | Original process seconds | Work |
| --- | ---: | --- |
| Prior collection | 197.3103 | 18 paths; 4,816 teacher calls and native steps |
| Native metadata bridge | 0.9389 | Source, runtime and collector provenance checks; no numerical decoding |
| Reanalysis evaluation | 16.8836 | Nine checkpoint decodes; six model constructions; three caches; 72 views |
| Independent audit | 9.8609 | Exact saved-estimator replay and independent scalar metrics; no neural calls |

These are instrumented process times, not a matched algorithm-latency benchmark.
Planning, engineering and five real-process handoff checks are recorded separately
in the [qualification evidence](../output/otto-residual-reanalysis-engineering-v1/handoff-01/receipt.json).
The repaired implementation passed 413 fabricated tests, lint and the fixed
capacity check. The first qualification attempt's lint failure remains preserved.

The [official closure](../output/otto-residual-reanalysis-v1/dev-closure-01.json)
joins the original producer and auditor. Both exited normally with reaped workers
and absent process groups. All 187 bound sources, including the original 177,
remain unchanged. A separate metadata/scalar review confirmed the complete
72-view roster, ratio selection, 13 conditions and covariance contrast.

The archive preserves all numerical outputs, original process evidence, source
bytes, the original development collection and all nine supplied checkpoints.
Its 374 members passed an exact opaque-byte roundtrip. Installed environments,
native teacher artifacts and earlier training-lineage dependencies remain
external; this is not a self-contained native reproduction.

## Research implication

Stop expanding posterior complexity for this fixed representation. The unresolved
question is what ordinary joint training changes in the base predictor or its
representation. The [proposed readout/state ablation](otto-readout-state-ablation-proposal.md)
isolates those changes before adding another memory mechanism. It is not
registered or executed. This result alone does not identify a particular
representation defect or justify a world-model claim.

This reanalysis closes at DEV. Its old reserved confirmation cohort and the
earlier study's TEST remain unused. No confirmation, autonomous advantage,
connectome advantage or ICLR-level novelty is established.
