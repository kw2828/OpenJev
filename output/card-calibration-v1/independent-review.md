# Independent saved-results review

Validation passed on the first audit attempt. Scientific continuation remains **FAIL, 11/12**; the original architecture comparison remains **FAIL, 1/6**.

The separate NumPy/stdlib checker imports no numerical fitter, reporter, model or environment. It verified the hashes and exact membership of 55 calibration payloads and 7,226 evaluation payloads, recalculated all 56 native rows and 3,584 games, reconstructed all 64 public deck layouts, and independently computed the 1,152 baseline-history probability scores and all continuation criteria. This validates saved evidence rather than reproducing model forwards or native environments. Scalar optimality and C-action replay were checked by the separate frozen reporter.

The sole failed condition is the GRU family mean return change of -0.01171875, below the allowed -0.01. Overall mean return increases by 0.2280982905982906. Mean NLL falls from 0.6969808060540342 to 0.4440193883430516; Brier falls from 0.22155283343280038 to 0.1620688982488979. The hard condition retains 63,915 infinite-NLL query errors.

A separate claim check found the README and results document consistent with the report. All 15 associative-memory fits finish 64/64 games under both temperature and hard conditions, each totaling 960/960. The three GRU fits remain at zero completion. The first fixed replay finishes under all three conditions in 94, 84 and 82 actions; it is not an outcome-selected example.

Root visually inspected the rendered chart and the middle/final GIF frames. The GIF has 95 frames at 160 ms per frame, with actual public observations. Its full file is embedded in the README.

- Independent checker SHA-256: `7fefc640a20193fa6dddf429ab28d9a2018378b5a532f8701f76977b6a4b516d`
- Independent completion SHA-256: `947c15950b8327a7a6ff2339e75a0e19145f55e8cb68a1c0ceb58ed4349c5435`
- Frozen report receipt SHA-256: `002de74906dddb04d2f1b33fc6b84b677042f520c7097245d343bda99ba89a8a`

No new model, native environment or scalar-fitting calls occurred during these reviews.
