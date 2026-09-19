# Post hoc CV16 diagnostic

320 new deterministic reference forecasts; zero neural/native calls or fitting. The original49-check gate is unchanged.

| Panel | Predictor | Normalized MSE | Pooled3D position RMSE(m) |
|---|---|---:|---:|
| test_sin | CV16 | 6.147801107 | 0.089104024 |
| test_sin | none | 9.765055007 | 0.296066679 |
| test_sin | bias | 9.675919567 | 0.114790672 |
| test_sin | public | 9.254396383 | 0.083591729 |
| test_sin | latent | 4.687418819 | 0.073372846 |
| test_sin | history16 | 9.632539066 | 0.293489563 |
| test_sin | ridge16 | 22.275491037 | 0.251616156 |
| test_sin | hold_last | 9.785579307 | 0.583865394 |
| test_zigzag | CV16 | 242293.222497190 | 0.148361710 |
| test_zigzag | none | 161253.166609401 | 0.512503899 |
| test_zigzag | bias | 232045.461125826 | 0.199726919 |
| test_zigzag | public | 256525.751806394 | 0.166616049 |
| test_zigzag | latent | 205291.352409220 | 0.150997118 |
| test_zigzag | history16 | 161218.600964225 | 0.501935683 |
| test_zigzag | ridge16 | 225788.884283558 | 13.210029236 |
| test_zigzag | hold_last | 161502.544888286 | 0.496740349 |

CV16 uses only context endpoints16/31. No fitted parameters, future actions, target feedback, or orientation projection.
New post hoc forecasts are not a saved-output-only audit. No architecture or primary-gate claim follows from this diagnostic.
