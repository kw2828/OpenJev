# Parallel Wiener-Hammerstein data adapter

**Implemented with fabricated-array qualification only.** No ParWH measurement
archive was downloaded or decoded for this work. This note fixes adapter
semantics, not a training or benchmark protocol. ParWH cannot test the proposed
[adaptive correction mechanism](predictive-state-routing-design.md) because its
single output makes block selection constant at fixed weights.

## Source and rights

The [authors' benchmark page](https://www.nonlinearbenchmark.org/benchmarks/parallel-wiener-hammerstein)
describes a real circuit with parallel nonlinear branches. The
[dataset, version 1](https://doi.org/10.4121/12950081) is licensed CC BY-SA 4.0
according to its DataCite metadata. Root repository MIT terms do not replace
those data terms. No source measurements are redistributed here.

The [identification paper](https://arxiv.org/abs/1708.06543) supplies the physical
setup and established modeling references. A dedicated parallel-Wiener-Hammerstein
identification method is a relevant strong baseline; beating a generic recurrent
network alone would not establish competitive performance.

Metadata and the official loader were inspected on 25 September 2026. The
loader was pinned to `4318c6b512b1083a46fb91cdae134bdb4dfffc11` in
[nonlinear_benchmarks](https://github.com/MaartenSchoukens/nonlinear_benchmarks/blob/4318c6b512b1083a46fb91cdae134bdb4dfffc11/nonlinear_benchmarks/benchmarks.py).
It documents native axes as `(sample, period, phase, amplitude)`. Our reader
preserves each period separately instead of flattening axes. This explicit
chronology contract makes no allegation about measured upstream behavior.

## Exact reader and split

[parwh_data.py](../src/openjev/research/parwh_data.py) requests only `uEst`, `yEst`,
`fs` and `amp` through SciPy's MAT variable whitelist. The file is snapshotted
before decoding; its hash is recorded from that same immutable snapshot.
Official test keys `uVal`, `yVal`,
`uValArr` and `yValArr` are never requested or returned.

Production estimation shape is exactly `(16384, 2, 20, 5)`. Each period is its
own chronological record: 200 records, 3,276,800 timepoints. The adapter
split assigns phase IDs 0-14 to FIT and 15-19 to DEV, keeping both periods and all
five amplitudes of a phase together. This yields 150 FIT and 50 DEV records.
Phase IDs are zero-based implementation indices.

The reader checks shapes, real float64 values and finiteness. It does not
interpolate, average periods, resample or delete observations. Normalization uses
population means and standard deviations from the complete FIT record roster
only; zero scales fail. DEV contributes no normalization statistics. Returned
record/window arrays own immutable backing storage. A separate fabricated-fixture
helper can reduce the sample axis, but cannot relax production geometry.

## Input timing and target separation

The native current input `u[k]` may influence current output `y[k]`, consistent
with the paper's nonlinear output-error equation. For start `s`, context `C`
and forecast horizon `H`, the adapter supplies:

```text
observed context:   y[s : s+C]
context inputs:     u[s+1 : s+C]
forecast inputs:    u[s+C : s+C+H]
forecast targets:   y[s+C : s+C+H]
```

The initializer sees `y[s]` only, so `u[s]` is omitted for every paired model.
There are exactly `C-1` transition inputs. The first forecast input predicts the
first target at the same native index. Later inputs cannot influence an earlier
forecast. The predictor receives the context and inputs through `public_inputs()`;
targets and original recordings are excluded from that interface. Windows may
not cross period boundaries.

Defaults are `C=50`, `H=256`. Those are custom pilot windows, not a reproduction
of an official benchmark score. Any future empirical admission must separately
freeze preprocessing, sampling, optimization and evaluation, then retain complete
predictions and costs. Schema qualification alone establishes no forecasting gain.
