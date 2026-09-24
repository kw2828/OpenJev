# Observation-operator component qualification

**24 fabricated tests, lint and all 16 geometry rollouts pass. No empirical
architecture result exists.** The component predicts through an observation gap
by summing its possible observation updates. An untied control can learn that
gap transition independently. Both start with the same functions within
float64 rounding.

The tests use an independent two-state rational example, enumerate observation
branches through three steps, check absorbing found outcomes and prevent future
observations from affecting earlier forecasts. They also verify combined
observed/blind gradients and an Adam update for both versions. A separate test
confirms that blind-only training leaves the untied observation operators
unchanged. Geometry checks cover batches of 1 and 32, nine prefix rows, and
horizons of 1 and 8.

| Version | Parameters | Parameter bytes | Fixed buffer bytes | Carried-state bytes per row |
|---|---:|---:|---:|---:|
| Tied observation marginal | 8,778 | 49,728 | 8 | 112 |
| Untied blind transition | 9,618 | 56,448 | 8 | 112 |

The prefix encoder uses float32; operators, mass and readout use float64.
Equal carry storage does not establish equal capacity, total memory or speed.
The geometry run is an inference-shape check, not a training-budget benchmark.

The first qualification passed its 24 tests but stopped on a lint error in the
wrapper's exception handler. The failed receipt and exact wrapper source are
retained. A separately recorded second attempt passed after narrowing that
exception handler. Model and test sources did not change between attempts.
Neither attempt decoded empirical data or changed the separately running
conditional-label experiment's registered inputs.

- [Component specification](otto-observation-operator-component.md)
- [Mechanism, related work and required controls](otto-predictive-moment-mechanism-draft.md)
- [Passing qualification receipt](../output/otto-observation-operator-component-v1/engineering-02/receipt.json)
- [Geometry and storage records](../output/otto-observation-operator-component-v1/engineering-02/geometry.json)
- [Preserved failed qualification](../output/otto-observation-operator-component-v1/engineering-01/receipt.json)

The next scientific comparison needs fresh data, matched observation inputs and
supervision, a GRU control and the untied operator control. Internal probability
consistency alone cannot establish better decisions, calibration, biological
learning or novelty.
