# Small nonlinear phase memory on Silverbox

**DO_NOT_ADVANCE_PHASE_MECHANISM: 3/21 frozen conditions passed.**

This is an internal development comparison on one measured electronic oscillator. Official benchmark TEST outputs remain numerically unparsed. It is not an official benchmark score, a new-architecture claim, or evidence about biological memory or control performance.

![Every trained seed and classical reference on both internal DEV sequences](benchmark.png)

FIT contains 43,296 samples. All 15 final fits completed before either 8,192-sample DEV sequence was loaded. Each prediction starts from zero state and uses inputs only, including the common 512-sample warmup; the remaining 7,680 samples are scored. Training uses 2,048 updates per fit with the same per-seed 32-window batches, length 256 and 64-sample loss burn. Measured outputs enter FIT losses and AR2 initialization, but never a scored rollout or DEV checkpoint selection.

Each phase model has four recurrent state scalars. The energy mechanism changes rotation using previous state energy; the 18-parameter nonlinear-readout control changes only the output map, while fixed phase has 14 parameters. All common initial tensors are paired and the three phase models initially implement the same function. GRU16 and the classical controls are not parameter- or compute-matched.

| Model / seed | DEV A RMSE mV | DEV A MAE mV | DEV B RMSE mV | DEV B MAE mV | Loop seconds | Logical bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed phase / 7301 | 7.12641 | 4.73355 | 5.25769 | 4.23545 | 31.4461 | 104 |
| Fixed phase / 7302 | 7.19067 | 4.92199 | 5.43865 | 4.41838 | 30.27 | 104 |
| Fixed phase / 7303 | 7.13914 | 4.74026 | 5.26101 | 4.23175 | 30.3051 | 104 |
| Energy phase / 7301 | 7.27997 | 5.22676 | 5.75406 | 4.73721 | 43.7357 | 120 |
| Energy phase / 7302 | 7.24238 | 4.99368 | 5.49792 | 4.47103 | 42.0678 | 120 |
| Energy phase / 7303 | 7.16979 | 4.6495 | 5.16294 | 4.10305 | 42.7188 | 120 |
| Nonlinear readout / 7301 | 6.32485 | 4.34854 | 4.83638 | 3.91003 | 41.5222 | 120 |
| Nonlinear readout / 7302 | 6.34921 | 4.27429 | 4.75837 | 3.81953 | 40.3883 | 120 |
| Nonlinear readout / 7303 | 6.38135 | 4.21995 | 4.70049 | 3.75573 | 41.3055 | 120 |
| GRU16 / 7301 | 1.0145 | 0.711253 | 0.834339 | 0.648414 | 20.3467 | 3,812 |
| GRU16 / 7302 | 0.855016 | 0.60697 | 0.718158 | 0.54496 | 20.1309 | 3,812 |
| GRU16 / 7303 | 0.761986 | 0.50014 | 0.629778 | 0.45046 | 20.5215 | 3,812 |
| Refined cubic AR2 / 7301 | 0.926387 | 0.734286 | 0.920657 | 0.741148 | 10.9662 | 72 |
| Refined cubic AR2 / 7302 | 0.954202 | 0.755273 | 0.950785 | 0.764256 | 11.1521 | 72 |
| Refined cubic AR2 / 7303 | 0.959373 | 0.759146 | 0.953017 | 0.765765 | 11.2381 | 72 |
| Static cubic | 54.0615 | 43.0494 | 53.4333 | 43.0397 | n/a | 64 |
| FIR128 | 7.17472 | 4.69036 | 5.21441 | 4.15273 | n/a | 2,080 |
| FIR512 | 7.20818 | 4.75983 | 5.40867 | 4.31035 | n/a | 8,224 |
| Frozen cubic AR2 | 3.29922 | 2.49667 | 3.05041 | 2.37729 | n/a | 72 |

Logical bytes include deployed parameters, recurrent state or FIR input queues, and 32 bytes of FIT normalizers. They exclude Python/native workspace, optimizer and training/audit arrays. Frozen AR2 stores its original float64 ridge coefficient file but simulates with float32 coefficients; the logical count uses that actual inference dtype. The figures do not establish deployment latency or a matched-compute advantage.

Loop seconds begin after model/Adam construction and FIT tensor conversion, and include the optimizer loop plus final Adam-slot extraction. They exclude checkpoint serialization. They are not inference timings. AR2 has a predeclared smaller learning rate of 0.0001; phase/GRU models use 0.003. The frozen ridge AR2 remains a separate comparator, so refinement cannot conceal a stronger classical initialization.

| Frozen condition | Result |
| --- | --- |
| `all_completed_predictions_finite` | PASS |
| `dev_a_mean_energy_improves_fixed_phase_10pct` | FAIL |
| `dev_a_seed7301_energy_not_worse_fixed_phase` | FAIL |
| `dev_a_seed7302_energy_not_worse_fixed_phase` | FAIL |
| `dev_a_seed7303_energy_not_worse_fixed_phase` | FAIL |
| `dev_a_mean_energy_improves_nonlinear_readout_10pct` | FAIL |
| `dev_a_seed7301_energy_not_worse_nonlinear_readout` | FAIL |
| `dev_a_seed7302_energy_not_worse_nonlinear_readout` | FAIL |
| `dev_a_seed7303_energy_not_worse_nonlinear_readout` | FAIL |
| `dev_a_within_5pct_best_conventional` | FAIL |
| `dev_a_competent_below_10pct_fit_std` | FAIL |
| `dev_b_mean_energy_improves_fixed_phase_10pct` | FAIL |
| `dev_b_seed7301_energy_not_worse_fixed_phase` | FAIL |
| `dev_b_seed7302_energy_not_worse_fixed_phase` | FAIL |
| `dev_b_seed7303_energy_not_worse_fixed_phase` | PASS |
| `dev_b_mean_energy_improves_nonlinear_readout_10pct` | FAIL |
| `dev_b_seed7301_energy_not_worse_nonlinear_readout` | FAIL |
| `dev_b_seed7302_energy_not_worse_nonlinear_readout` | FAIL |
| `dev_b_seed7303_energy_not_worse_nonlinear_readout` | FAIL |
| `dev_b_within_5pct_best_conventional` | FAIL |
| `dev_b_competent_below_10pct_fit_std` | PASS |

The rule requires 10% mean improvements over both phase controls on both DEV sequences, no worse result for any paired seed, competence below 10% of FIT output standard deviation, and a result within 5% of the strongest conventional comparator. Three optimization seeds share the same physical observations; they are not independent datasets, and no significance claim is made. A pass only supports a separately frozen next experiment. A failed condition cannot be rescued by choosing a favorable seed, DEV sequence or comparator.

The independent audit checked all 135 manifest-listed files, decoded 64 NPZ and 41 NPY files containing 504 saved arrays, and replayed 38 final predictions using independent NumPy equations. It checked all 30 initial/final model files, 15 Adam files and four ridge solutions. For each ridge solution it checked a regularized normal-equation backward error bounded by 64 times gamma_d, where gamma_d=d*u/(1-d*u), d is coefficient count and u is float64 unit roundoff. Saved-versus-independent predictions were compared on FIT and both DEV sequences (12 comparisons); coefficient differences remain diagnostics because ill-conditioned directions can have different coefficients but nearly identical predictions. All four certificates are retained in the plotted-values JSON. It did not replay 30,720 optimizer updates. Its fixed normalized per-point tolerance is rtol=atol=1e-4; this is an engineering guard, not a proof of incremental stability or arbitrary-parameter floating-point equivalence. The cell has a bounded-input, bounded-state argument, which does not imply bounded training gradients.

Original qualification process: 4.005217 seconds. Original run: 442.443785 seconds. Original independent audit: 2.264795 seconds. These are whole-process durations. This presentation helper decoded no scientific arrays and made no model, optimizer or numerical audit calls.

Registration SHA256: `258725c1db149c76ec54070e0ec236367489a096554f658b2fe8a9e5066cc2ce`. Pre-fit source commit: `b1395f4e924d5bc19bc3a12cdb8104a49a51fa48`.

[All plotted values, costs and conditions](plotted-values.json) | [PDF figure](benchmark.pdf) | [Presentation provenance](plot-receipt.json)

Source: [Silverbox benchmark](https://www.nonlinearbenchmark.org/benchmarks/silverbox). Related mechanisms include [coRNN](https://arxiv.org/abs/2010.00951), [ReLiNet](https://www.ijcai.org/proceedings/2023/0385.pdf), and [polynomial nonlinear state-space identification](https://doi.org/10.1016/j.automatica.2010.01.001). The official initialization and TEST protocol differs from this internal comparison. Raw measurements and retained FIT/DEV arrays stay local because explicit redistribution permission has not been established; software licensing is a separate question.
