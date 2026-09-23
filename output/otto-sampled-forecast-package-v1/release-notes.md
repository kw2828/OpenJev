Complete sampled-annotation forecast experiment: 90 full trajectories, 12 final model fits, and an agreeing independent saved-output audit.

The proposed residual GRU fails its fixed continuation screen, **41/45**. All four failed checks compare it with an ordinary direct GRU, whose mean teacher agreement and raw score gap are better in both settings.

| Model | Length 3 agreement | Length 4 agreement |
|---|---:|---:|
| Hold last scores | 30.24% | 33.27% |
| Proposed residual GRU, mean of 3 fits | 71.53% | 62.74% |
| Ordinary direct GRU, mean of 3 fits | 76.54% | 64.36% |

These are forecasts on collector-generated paths. This result establishes no autonomous-control benefit, measured deployment savings or architectural novelty. The failed rule remains unchanged and does not promote a selected control or seed.

The archive preserves all public trajectories and teacher-call records, sampled training arrays and weights, all twelve checkpoints, thirteen validation prediction files, complete metrics, 45 conditions, original worker/supervisor records, source qualifications and inherited evidence. Both PNG/SVG figures and complete CSV tables are included. See the manifest and RESTORE.md for the exact inventory and external native-runtime/weight dependencies.

The preceding incomplete collection and failed exact-cache screen remain preserved. No old allocation was resumed or fitted.
