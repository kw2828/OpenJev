# Structured robot transition: complete saved results

Development on the same two previously exposed recordings. Both rates are retained in the complete table. Selected-rate dots show all three individual fits; accuracy diamonds are arithmetic means, not an ensemble. 32 observed steps followed by 128 forecast steps (12.8 s), conditional on realized measured torque. Each forecast uses only preceding torque inputs. Internal CONFIRM and official TEST remain closed. Legacy fits and ridge banks are frozen parent artifacts; all latency probes are current-host. Different model sizes and precision; no novelty, control-safety, Rust or official benchmark claim.

Saved outcome: **DO_NOT_ADVANCE_STRUCTURED_TRANSITION**, 23/61 conditions.

## Every candidate and reference

All160 rows are retained. H64 is descriptive; two-rate selection and the rule use pooled DEV H128. Both causal ridge penalties are visible.

| Recording | Model | Rate | Seed | Horizon | Status | Standardized RMSE | Physical RMSE (deg) | Per-joint RMSE (deg) | Error |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| 21H_54M | Householder | 0.001 | 8101 | 64 | PASS | 0.588962 | 16.6121 | 12.4261, 3.57187, 13.2459, 26.7718, 21.5826, 11.4284 |  |
| 21H_54M | Householder | 0.001 | 8101 | 128 | PASS | 0.756956 | 21.7203 | 15.7607, 4.24838, 16.711, 34.7494, 27.554, 17.8374 |  |
| 21H_54M | Bounded dense | 0.001 | 8101 | 64 | PASS | 0.526761 | 15.071 | 7.97611, 3.54818, 11.8637, 24.1952, 19.6518, 13.2008 |  |
| 21H_54M | Bounded dense | 0.001 | 8101 | 128 | PASS | 0.668641 | 19.452 | 9.92339, 4.09209, 14.0095, 31.7778, 25.3465, 17.5074 |  |
| 21H_54M | Unbounded dense | 0.001 | 8101 | 64 | PASS | 0.529701 | 14.6402 | 7.90586, 4.06536, 12.9277, 23.773, 19.681, 9.34624 |  |
| 21H_54M | Unbounded dense | 0.001 | 8101 | 128 | PASS | 0.647494 | 18.7129 | 10.5398, 4.46082, 13.4065, 32.4055, 23.8468, 13.0968 |  |
| 21H_54M | Bounded dense + MLP | 0.001 | 8101 | 64 | PASS | 0.565003 | 16.5028 | 7.03596, 4.33537, 13.9418, 29.0579, 19.0795, 12.7663 |  |
| 21H_54M | Bounded dense + MLP | 0.001 | 8101 | 128 | PASS | 0.646556 | 19.4643 | 9.24222, 4.59126, 13.7769, 35.832, 21.7599, 14.8134 |  |
| 21H_54M | GRU32 residual | 0.001 | 8101 | 64 | PASS | 0.621029 | 17.7526 | 12.8772, 4.95138, 15.0947, 23.8605, 19.6491, 22.745 |  |
| 21H_54M | GRU32 residual | 0.001 | 8101 | 128 | PASS | 0.726081 | 21.3665 | 16.055, 5.26703, 16.7994, 29.8219, 22.1598, 28.1255 |  |
| 21H_54M | Householder | 0.003 | 8101 | 64 | PASS | 0.570147 | 16.2517 | 12.6599, 3.37515, 13.0072, 26.2774, 20.1323, 12.167 |  |
| 21H_54M | Householder | 0.003 | 8101 | 128 | PASS | 0.682588 | 20.2942 | 14.3544, 3.79227, 15.352, 34.9572, 22.7246, 16.6313 |  |
| 21H_54M | Bounded dense | 0.003 | 8101 | 64 | PASS | 0.498904 | 14.1863 | 8.56349, 3.20608, 12.5769, 23.2632, 17.856, 10.2815 |  |
| 21H_54M | Bounded dense | 0.003 | 8101 | 128 | PASS | 0.57881 | 17.0303 | 9.97674, 3.60631, 12.5238, 29.4031, 20.5905, 13.5011 |  |
| 21H_54M | Unbounded dense | 0.003 | 8101 | 64 | PASS | 0.580113 | 17.6083 | 6.86614, 4.28494, 11.928, 33.0033, 19.5473, 13.4616 |  |
| 21H_54M | Unbounded dense | 0.003 | 8101 | 128 | PASS | 0.697515 | 22.471 | 9.43034, 4.71715, 13.3499, 45.7203, 20.2139, 15.5346 |  |
| 21H_54M | Bounded dense + MLP | 0.003 | 8101 | 64 | PASS | 0.584582 | 17.7709 | 9.27191, 4.15418, 12.1282, 32.7406, 18.9811, 14.57 |  |
| 21H_54M | Bounded dense + MLP | 0.003 | 8101 | 128 | PASS | 0.676109 | 21.3325 | 11.6863, 4.49762, 11.9753, 41.1328, 20.6934, 17.6105 |  |
| 21H_54M | GRU32 residual | 0.003 | 8101 | 64 | PASS | 0.632467 | 17.9166 | 16.1159, 3.77955, 13.6704, 25.0657, 21.8054, 19.0103 |  |
| 21H_54M | GRU32 residual | 0.003 | 8101 | 128 | PASS | 0.719177 | 20.6618 | 18.9225, 4.34463, 13.8835, 31.7259, 24.4143, 19.7279 |  |
| 21H_54M | Householder | 0.001 | 8102 | 64 | PASS | 0.568374 | 16.2956 | 11.8688, 4.24541, 12.5533, 27.0306, 19.0784, 13.4966 |  |
| 21H_54M | Householder | 0.001 | 8102 | 128 | PASS | 0.654792 | 18.9014 | 13.9595, 4.2618, 14, 31.2328, 22.7943, 15.4748 |  |
| 21H_54M | Bounded dense | 0.001 | 8102 | 64 | PASS | 0.50995 | 14.7222 | 8.1731, 3.47996, 13.0079, 25.2392, 17.4879, 10.4639 |  |
| 21H_54M | Bounded dense | 0.001 | 8102 | 128 | PASS | 0.600547 | 17.1321 | 10.3996, 3.73263, 13.2872, 28.8575, 22.566, 10.9744 |  |
| 21H_54M | Unbounded dense | 0.001 | 8102 | 64 | PASS | 0.609525 | 17.274 | 10.4221, 5.95896, 14.2044, 29.9524, 18.6706, 14.0968 |  |
| 21H_54M | Unbounded dense | 0.001 | 8102 | 128 | PASS | 0.685994 | 19.919 | 13.4485, 6.19452, 15.7622, 34.5576, 20.0792, 17.7631 |  |
| 21H_54M | Bounded dense + MLP | 0.001 | 8102 | 64 | PASS | 0.52896 | 15.7027 | 8.23652, 3.89589, 12.6302, 28.1645, 17.0915, 12.3105 |  |
| 21H_54M | Bounded dense + MLP | 0.001 | 8102 | 128 | PASS | 0.594783 | 18.0202 | 10.335, 3.91041, 12.9746, 32.971, 19.2516, 14.1495 |  |
| 21H_54M | GRU32 residual | 0.001 | 8102 | 64 | PASS | 0.600895 | 17.5552 | 11.2942, 4.75718, 11.3977, 23.4352, 20.5654, 24.4308 |  |
| 21H_54M | GRU32 residual | 0.001 | 8102 | 128 | PASS | 0.729859 | 21.618 | 13.1461, 4.78545, 14.169, 30.0687, 25.9577, 28.8031 |  |
| 21H_54M | Householder | 0.003 | 8102 | 64 | PASS | 0.584045 | 16.6345 | 17.2717, 3.75253, 11.3726, 27.6697, 18.3324, 10.8079 |  |
| 21H_54M | Householder | 0.003 | 8102 | 128 | PASS | 0.669856 | 19.641 | 16.2077, 3.82875, 13.1281, 33.0235, 22.7607, 16.0098 |  |
| 21H_54M | Bounded dense | 0.003 | 8102 | 64 | PASS | 0.577858 | 17.2711 | 9.96965, 4.58632, 12.7707, 28.9022, 17.941, 18.6818 |  |
| 21H_54M | Bounded dense | 0.003 | 8102 | 128 | PASS | 0.645541 | 19.5141 | 11.3351, 4.97906, 12.9709, 32.4311, 20.3622, 22.2907 |  |
| 21H_54M | Unbounded dense | 0.003 | 8102 | 64 | PASS | 0.533886 | 15.49 | 9.53147, 4.30046, 12.6435, 26.6401, 17.1026, 12.9705 |  |
| 21H_54M | Unbounded dense | 0.003 | 8102 | 128 | PASS | 0.607108 | 18.118 | 11.5649, 4.726, 12.7241, 31.3454, 18.9663, 17.5878 |  |
| 21H_54M | Bounded dense + MLP | 0.003 | 8102 | 64 | PASS | 0.56451 | 16.4945 | 8.77556, 4.37765, 15.1714, 29.6121, 17.3665, 11.2957 |  |
| 21H_54M | Bounded dense + MLP | 0.003 | 8102 | 128 | PASS | 0.637895 | 19.4188 | 9.85132, 4.45459, 15.474, 36.595, 19.4182, 13.7821 |  |
| 21H_54M | GRU32 residual | 0.003 | 8102 | 64 | PASS | 0.604414 | 18.0611 | 13.6913, 4.383, 13.3341, 25.5906, 17.8066, 24.5113 |  |
| 21H_54M | GRU32 residual | 0.003 | 8102 | 128 | PASS | 0.731339 | 21.5541 | 16.3737, 4.87746, 16.9487, 28.8175, 22.8782, 29.2314 |  |
| 21H_54M | Householder | 0.001 | 8103 | 64 | PASS | 0.550925 | 15.3462 | 9.48952, 3.61734, 13.5692, 24.5821, 20.5805, 9.89675 |  |
| 21H_54M | Householder | 0.001 | 8103 | 128 | PASS | 0.67326 | 19.001 | 11.4883, 3.8757, 15.6203, 30.2731, 25.8878, 13.733 |  |
| 21H_54M | Bounded dense | 0.001 | 8103 | 64 | PASS | 0.552468 | 16.0846 | 7.90955, 4.1572, 12.3278, 28.1113, 19.3966, 12.4095 |  |
| 21H_54M | Bounded dense | 0.001 | 8103 | 128 | PASS | 0.641108 | 19.0454 | 9.62592, 4.38948, 13.525, 34.565, 22.4643, 13.4958 |  |
| 21H_54M | Unbounded dense | 0.001 | 8103 | 64 | PASS | 0.526367 | 14.843 | 6.8372, 4.04207, 12.81, 23.8536, 19.2137, 12.5117 |  |
| 21H_54M | Unbounded dense | 0.001 | 8103 | 128 | PASS | 0.629915 | 18.3192 | 8.3921, 4.36983, 14.2869, 31.1265, 22.8803, 15.0845 |  |
| 21H_54M | Bounded dense + MLP | 0.001 | 8103 | 64 | PASS | 0.522303 | 15.4464 | 7.24851, 4.00093, 11.4313, 26.0956, 17.793, 15.3217 |  |
| 21H_54M | Bounded dense + MLP | 0.001 | 8103 | 128 | PASS | 0.613607 | 18.6648 | 8.93173, 4.24088, 12.2722, 32.7018, 20.7171, 18.5274 |  |
| 21H_54M | GRU32 residual | 0.001 | 8103 | 64 | PASS | 0.633333 | 17.6622 | 13.9196, 5.42396, 12.5868, 24.1944, 21.7148, 20.814 |  |
| 21H_54M | GRU32 residual | 0.001 | 8103 | 128 | PASS | 0.736532 | 20.7522 | 16.0154, 5.78958, 14.1504, 28.5276, 26.0512, 24.5192 |  |
| 21H_54M | Householder | 0.003 | 8103 | 64 | PASS | 0.55649 | 15.0347 | 13.6533, 4.007, 11.9836, 21.8724, 20.3042, 10.9325 |  |
| 21H_54M | Householder | 0.003 | 8103 | 128 | PASS | 0.712273 | 19.5379 | 15.0035, 4.5089, 14.605, 26.8391, 27.9413, 18.1816 |  |
| 21H_54M | Bounded dense | 0.003 | 8103 | 64 | PASS | 0.594425 | 17.8921 | 10.3689, 3.85054, 11.8221, 32.1034, 20.3162, 14.6729 |  |
| 21H_54M | Bounded dense | 0.003 | 8103 | 128 | PASS | 0.666515 | 20.0955 | 11.8855, 4.24717, 13.4867, 34.2226, 22.5879, 20.0094 |  |
| 21H_54M | Unbounded dense | 0.003 | 8103 | 64 | PASS | 0.546035 | 15.4916 | 8.79434, 3.70667, 11.9819, 24.2185, 20.6283, 13.9007 |  |
| 21H_54M | Unbounded dense | 0.003 | 8103 | 128 | PASS | 0.644397 | 18.8586 | 10.5957, 3.9943, 12.8868, 31.6804, 23.9877, 16.1409 |  |
| 21H_54M | Bounded dense + MLP | 0.003 | 8103 | 64 | PASS | 0.570523 | 17.045 | 8.09289, 3.80269, 13.182, 31.4016, 19.3951, 11.2799 |  |
| 21H_54M | Bounded dense + MLP | 0.003 | 8103 | 128 | PASS | 0.639334 | 19.3248 | 10.0058, 3.7516, 13.86, 35.1556, 22.0519, 14.5669 |  |
| 21H_54M | GRU32 residual | 0.003 | 8103 | 64 | PASS | 0.641834 | 18.9613 | 11.3306, 5.21776, 10.2278, 31.2694, 22.2804, 20.5619 |  |
| 21H_54M | GRU32 residual | 0.003 | 8103 | 128 | PASS | 0.777481 | 23.5199 | 14.5145, 5.33618, 10.4228, 41.3938, 27.5386, 22.3498 |  |
| 21H_54M | Legacy instant (cached) | 0.001 | 8101 | 64 | PASS | 0.56324 | 16.9653 | 6.67805, 4.02551, 13.9768, 32.0966, 18.0975, 10.6332 |  |
| 21H_54M | Legacy instant (cached) | 0.001 | 8101 | 128 | PASS | 0.65997 | 20.4468 | 8.94, 4.42939, 14.7836, 39.6818, 20.6215, 13.7996 |  |
| 21H_54M | Legacy instant (cached) | 0.003 | 8101 | 64 | PASS | 0.575331 | 16.8546 | 8.24717, 3.76089, 12.069, 29.5873, 21.2465, 12.2405 |  |
| 21H_54M | Legacy instant (cached) | 0.003 | 8101 | 128 | PASS | 0.642299 | 18.876 | 9.22483, 4.28916, 11.9004, 32.4879, 24.2005, 15.8613 |  |
| 21H_54M | Legacy instant (cached) | 0.001 | 8102 | 64 | PASS | 0.514277 | 14.5716 | 8.30987, 4.05307, 11.6938, 22.5592, 18.3277, 14.385 |  |
| 21H_54M | Legacy instant (cached) | 0.001 | 8102 | 128 | PASS | 0.605261 | 17.864 | 10.8638, 4.42171, 12.2224, 29.6511, 20.6557, 17.9421 |  |
| 21H_54M | Legacy instant (cached) | 0.003 | 8102 | 64 | PASS | 0.564161 | 17.1923 | 12.3704, 3.31926, 11.3339, 30.7248, 17.7581, 14.8855 |  |
| 21H_54M | Legacy instant (cached) | 0.003 | 8102 | 128 | PASS | 0.632774 | 19.2623 | 13.3839, 3.7759, 11.6092, 34.5008, 20.7915, 16.5972 |  |
| 21H_54M | Legacy instant (cached) | 0.001 | 8103 | 64 | PASS | 0.515074 | 15.1888 | 6.42573, 3.48004, 13.2965, 25.8463, 17.4322, 13.4936 |  |
| 21H_54M | Legacy instant (cached) | 0.001 | 8103 | 128 | PASS | 0.612232 | 18.2915 | 8.12166, 3.78753, 14.1794, 31.9979, 21.4707, 15.5325 |  |
| 21H_54M | Legacy instant (cached) | 0.003 | 8103 | 64 | PASS | 0.571113 | 16.7186 | 8.92731, 3.74431, 13.919, 28.3508, 19.7328, 14.0166 |  |
| 21H_54M | Legacy instant (cached) | 0.003 | 8103 | 128 | PASS | 0.648573 | 19.4742 | 10.1487, 4.22734, 14.2676, 33.9105, 22.0256, 17.776 |  |
| 21H_54M | Causal ridge (1) | n/a | None | 64 | PASS | 0.559915 | 16.2719 | 10.4995, 5.65965, 12.582, 27.7244, 15.1837, 16.9964 |  |
| 21H_54M | Causal ridge (1) | n/a | None | 128 | PASS | 0.744225 | 21.9132 | 14.0532, 7.23159, 15.9085, 36.3675, 20.6251, 25.1049 |  |
| 21H_54M | Causal ridge (100) | n/a | None | 64 | PASS | 0.584402 | 17.3852 | 9.47897, 4.95741, 12.2164, 28.6224, 18.4429, 19.7593 |  |
| 21H_54M | Causal ridge (100) | n/a | None | 128 | PASS | 0.675447 | 20.2353 | 11.7595, 5.7122, 13.5981, 33.4459, 20.8447, 23.4061 |  |
| 21H_54M | Frozen linear AR2 | n/a | None | 64 | PASS | 0.730026 | 21.2211 | 10.7519, 5.33914, 23.0623, 28.0936, 20.8668, 28.3084 |  |
| 21H_54M | Frozen linear AR2 | n/a | None | 128 | PASS | 1.08777 | 30.8077 | 19.7813, 7.27011, 40.4915, 37.0243, 26.0793, 39.4976 |  |
| 21H_54M | Persistence | n/a | None | 64 | PASS | 1.13736 | 32.6754 | 20.8189, 10.4477, 29.6569, 49.0193, 32.2645, 39.2438 |  |
| 21H_54M | Persistence | n/a | None | 128 | PASS | 1.23294 | 35.4162 | 26.9324, 11.1036, 31.1057, 53.3282, 33.6346, 41.6469 |  |
| 22H_10M | Householder | 0.001 | 8101 | 64 | PASS | 0.638782 | 18.1163 | 13.5138, 5.79936, 13.7668, 31.2006, 19.9412, 13.8667 |  |
| 22H_10M | Householder | 0.001 | 8101 | 128 | PASS | 0.781211 | 22.5204 | 17.428, 6.0464, 15.9061, 36.6996, 25.635, 21.1114 |  |
| 22H_10M | Bounded dense | 0.001 | 8101 | 64 | PASS | 0.625723 | 17.4025 | 10.4154, 5.7092, 13.3423, 29.5042, 21.9302, 12.1058 |  |
| 22H_10M | Bounded dense | 0.001 | 8101 | 128 | PASS | 0.707726 | 20.1032 | 13.0973, 6.13041, 15.5437, 33.4234, 23.5433, 17.3979 |  |
| 22H_10M | Unbounded dense | 0.001 | 8101 | 64 | PASS | 0.644638 | 18.7731 | 7.96455, 4.98895, 12.7901, 32.5917, 23.6532, 15.5235 |  |
| 22H_10M | Unbounded dense | 0.001 | 8101 | 128 | PASS | 0.7085 | 20.9269 | 10.3076, 5.21536, 12.6229, 37.1547, 25.8387, 16.9324 |  |
| 22H_10M | Bounded dense + MLP | 0.001 | 8101 | 64 | PASS | 0.59081 | 17.1934 | 8.23679, 5.1576, 11.988, 29.6222, 20.1571, 15.8665 |  |
| 22H_10M | Bounded dense + MLP | 0.001 | 8101 | 128 | PASS | 0.673702 | 20.3342 | 11.2781, 5.38941, 12.4394, 35.5613, 21.7852, 20.753 |  |
| 22H_10M | GRU32 residual | 0.001 | 8101 | 64 | PASS | 0.720924 | 21.6809 | 12.722, 6.0561, 12.1103, 34.0208, 23.3099, 27.8283 |  |
| 22H_10M | GRU32 residual | 0.001 | 8101 | 128 | PASS | 0.851546 | 26.249 | 16.4884, 6.23687, 12.3297, 42.991, 27.5158, 32.6487 |  |
| 22H_10M | Householder | 0.003 | 8101 | 64 | PASS | 0.682624 | 20.1167 | 14.584, 6.01005, 14.125, 37.8707, 19.6508, 12.626 |  |
| 22H_10M | Householder | 0.003 | 8101 | 128 | PASS | 0.788774 | 23.5288 | 16.7195, 5.91055, 15.9081, 42.2733, 24.4919, 19.1626 |  |
| 22H_10M | Bounded dense | 0.003 | 8101 | 64 | PASS | 0.591939 | 16.8963 | 8.66945, 5.44198, 12.9506, 28.9483, 19.8125, 14.4871 |  |
| 22H_10M | Bounded dense | 0.003 | 8101 | 128 | PASS | 0.655627 | 18.4997 | 12.3094, 5.76479, 13.3903, 30.3875, 22.4534, 16.1812 |  |
| 22H_10M | Unbounded dense | 0.003 | 8101 | 64 | PASS | 0.642589 | 18.7058 | 10.8529, 6.39727, 14.6558, 33.2261, 18.1083, 17.1481 |  |
| 22H_10M | Unbounded dense | 0.003 | 8101 | 128 | PASS | 0.734903 | 22.3766 | 13.0754, 6.72846, 15.9589, 39.9501, 19.1682, 23.8731 |  |
| 22H_10M | Bounded dense + MLP | 0.003 | 8101 | 64 | PASS | 0.569649 | 16.5936 | 9.30493, 5.56208, 12.2512, 28.0598, 17.0256, 17.5285 |  |
| 22H_10M | Bounded dense + MLP | 0.003 | 8101 | 128 | PASS | 0.658802 | 19.7809 | 12.9405, 5.84305, 13.1103, 33.6243, 18.7946, 22.1448 |  |
| 22H_10M | GRU32 residual | 0.003 | 8101 | 64 | PASS | 0.737653 | 21.8881 | 14.5127, 5.56099, 13.1065, 36.1745, 24.822, 23.1619 |  |
| 22H_10M | GRU32 residual | 0.003 | 8101 | 128 | PASS | 0.799893 | 24.3922 | 17.4928, 5.55672, 11.8895, 42.1234, 25.872, 25.4538 |  |
| 22H_10M | Householder | 0.001 | 8102 | 64 | PASS | 0.683128 | 20.1598 | 14.0725, 5.80592, 15.1423, 37.0281, 19.7678, 14.6838 |  |
| 22H_10M | Householder | 0.001 | 8102 | 128 | PASS | 0.80271 | 23.9283 | 18.0447, 6.0228, 16.3062, 42.7128, 24.2386, 19.8922 |  |
| 22H_10M | Bounded dense | 0.001 | 8102 | 64 | PASS | 0.602257 | 16.9786 | 8.00179, 5.14593, 12.8397, 27.6397, 22.0179, 15.0174 |  |
| 22H_10M | Bounded dense | 0.001 | 8102 | 128 | PASS | 0.684597 | 19.6993 | 11.9503, 5.48683, 13.5086, 32.4306, 24.1658, 18.364 |  |
| 22H_10M | Unbounded dense | 0.001 | 8102 | 64 | PASS | 0.64808 | 19.1692 | 9.67976, 5.76616, 14.1033, 34.0656, 20.152, 17.6728 |  |
| 22H_10M | Unbounded dense | 0.001 | 8102 | 128 | PASS | 0.725201 | 21.676 | 12.6426, 6.60823, 14.9881, 38.9674, 21.084, 20.687 |  |
| 22H_10M | Bounded dense + MLP | 0.001 | 8102 | 64 | PASS | 0.61299 | 18.0698 | 9.86365, 4.88316, 12.9462, 30.2576, 20.4285, 18.3717 |  |
| 22H_10M | Bounded dense + MLP | 0.001 | 8102 | 128 | PASS | 0.694918 | 21.2234 | 12.7969, 5.08947, 12.6971, 35.6107, 22.1665, 24.3361 |  |
| 22H_10M | GRU32 residual | 0.001 | 8102 | 64 | PASS | 0.696526 | 20.7822 | 12.6032, 6.03208, 12.0299, 31.9513, 22.3178, 27.0642 |  |
| 22H_10M | GRU32 residual | 0.001 | 8102 | 128 | PASS | 0.790449 | 23.2249 | 16.8609, 6.16295, 14.364, 35.4431, 26.0404, 27.8113 |  |
| 22H_10M | Householder | 0.003 | 8102 | 64 | PASS | 0.687781 | 19.3321 | 15.4578, 6.33377, 14.8917, 34.5952, 21.1172, 9.93933 |  |
| 22H_10M | Householder | 0.003 | 8102 | 128 | PASS | 0.726599 | 21.2942 | 15.622, 6.09068, 14.6419, 38.7776, 22.2265, 15.08 |  |
| 22H_10M | Bounded dense | 0.003 | 8102 | 64 | PASS | 0.651257 | 19.276 | 11.4629, 5.77861, 12.9151, 31.6424, 20.2541, 22.0527 |  |
| 22H_10M | Bounded dense | 0.003 | 8102 | 128 | PASS | 0.74173 | 22.1881 | 15.3696, 6.36511, 14.1562, 34.1238, 21.904, 28.8532 |  |
| 22H_10M | Unbounded dense | 0.003 | 8102 | 64 | PASS | 0.605395 | 17.5401 | 9.05938, 5.5593, 14.3626, 30.2443, 18.7391, 16.1494 |  |
| 22H_10M | Unbounded dense | 0.003 | 8102 | 128 | PASS | 0.684968 | 20.4878 | 12.8466, 5.41683, 14.43, 34.7862, 21.1614, 21.401 |  |
| 22H_10M | Bounded dense + MLP | 0.003 | 8102 | 64 | PASS | 0.598442 | 18.1329 | 11.1605, 5.19586, 12.3058, 32.4393, 16.8612, 18.2545 |  |
| 22H_10M | Bounded dense + MLP | 0.003 | 8102 | 128 | PASS | 0.653715 | 20.3738 | 13.0738, 5.13153, 12.1425, 36.6, 18.02, 21.9447 |  |
| 22H_10M | GRU32 residual | 0.003 | 8102 | 64 | PASS | 0.76431 | 22.9932 | 12.9346, 5.8344, 14.7119, 38.927, 25.1328, 24.645 |  |
| 22H_10M | GRU32 residual | 0.003 | 8102 | 128 | PASS | 0.817338 | 25.0095 | 13.8949, 5.95797, 14.1153, 42.1684, 26.9484, 28.647 |  |
| 22H_10M | Householder | 0.001 | 8103 | 64 | PASS | 0.718865 | 20.5754 | 14.5281, 6.11797, 13.3755, 35.8276, 24.3782, 15.3225 |  |
| 22H_10M | Householder | 0.001 | 8103 | 128 | PASS | 0.82365 | 23.8899 | 17.6837, 6.13861, 15.503, 40.8466, 28.1034, 19.3746 |  |
| 22H_10M | Bounded dense | 0.001 | 8103 | 64 | PASS | 0.558376 | 16.0833 | 9.03722, 4.8843, 13.2997, 25.856, 17.8248, 16.8338 |  |
| 22H_10M | Bounded dense | 0.001 | 8103 | 128 | PASS | 0.624991 | 18.908 | 10.5493, 4.79806, 13.2851, 31.027, 19.4953, 22.1704 |  |
| 22H_10M | Unbounded dense | 0.001 | 8103 | 64 | PASS | 0.592164 | 16.9542 | 9.76737, 5.31649, 13.1338, 26.8866, 19.4366, 18.106 |  |
| 22H_10M | Unbounded dense | 0.001 | 8103 | 128 | PASS | 0.654883 | 19.4877 | 11.1087, 5.39655, 13.1323, 31.835, 21.0923, 22.2549 |  |
| 22H_10M | Bounded dense + MLP | 0.001 | 8103 | 64 | PASS | 0.588971 | 17.2147 | 9.34023, 5.66246, 12.4405, 30.2804, 17.908, 16.3218 |  |
| 22H_10M | Bounded dense + MLP | 0.001 | 8103 | 128 | PASS | 0.666784 | 20.0471 | 11.9768, 5.69761, 14.0736, 36.0858, 19.552, 18.7848 |  |
| 22H_10M | GRU32 residual | 0.001 | 8103 | 64 | PASS | 0.703977 | 21.2265 | 15.1821, 5.76424, 11.0164, 34.5981, 21.7334, 25.474 |  |
| 22H_10M | GRU32 residual | 0.001 | 8103 | 128 | PASS | 0.79226 | 23.8753 | 18.4228, 6.2254, 12.8465, 38.219, 23.9853, 29.0003 |  |
| 22H_10M | Householder | 0.003 | 8103 | 64 | PASS | 0.720662 | 20.6958 | 15.0182, 6.11466, 14.5409, 34.4964, 23.469, 18.8345 |  |
| 22H_10M | Householder | 0.003 | 8103 | 128 | PASS | 0.943293 | 28.7973 | 17.586, 6.52458, 16.9612, 44.4795, 30.9185, 37.4406 |  |
| 22H_10M | Bounded dense | 0.003 | 8103 | 64 | PASS | 0.6511 | 19.4013 | 14.7808, 5.31881, 13.1728, 34.0567, 18.6932, 18.1353 |  |
| 22H_10M | Bounded dense | 0.003 | 8103 | 128 | PASS | 0.727831 | 21.9731 | 16.7625, 5.7068, 14.4394, 36.5872, 20.5427, 24.7834 |  |
| 22H_10M | Unbounded dense | 0.003 | 8103 | 64 | PASS | 0.581147 | 16.4226 | 11.8383, 5.18093, 13.168, 26.7628, 18.423, 14.9051 |  |
| 22H_10M | Unbounded dense | 0.003 | 8103 | 128 | PASS | 0.656548 | 18.8113 | 14.5666, 5.38821, 13.4499, 29.1352, 21.2868, 19.9771 |  |
| 22H_10M | Bounded dense + MLP | 0.003 | 8103 | 64 | PASS | 0.618539 | 18.3011 | 12.2684, 5.03142, 13.0915, 31.5504, 19.0039, 17.4866 |  |
| 22H_10M | Bounded dense + MLP | 0.003 | 8103 | 128 | PASS | 0.659664 | 19.689 | 14.7815, 5.15563, 13.5116, 32.7648, 19.4259, 21.152 |  |
| 22H_10M | GRU32 residual | 0.003 | 8103 | 64 | PASS | 0.630417 | 18.1204 | 13.2445, 5.37538, 11.8788, 28.0972, 20.8603, 20.0017 |  |
| 22H_10M | GRU32 residual | 0.003 | 8103 | 128 | PASS | 0.758204 | 22.5063 | 15.5351, 5.94541, 13.5228, 35.2612, 24.7319, 26.919 |  |
| 22H_10M | Legacy instant (cached) | 0.001 | 8101 | 64 | PASS | 0.637721 | 18.8098 | 9.04651, 5.18593, 13.5674, 32.0806, 21.4383, 18.474 |  |
| 22H_10M | Legacy instant (cached) | 0.001 | 8101 | 128 | PASS | 0.700229 | 20.8666 | 12.3298, 5.32696, 14.0171, 33.5766, 23.1534, 23.9194 |  |
| 22H_10M | Legacy instant (cached) | 0.003 | 8101 | 64 | PASS | 0.668639 | 19.4381 | 9.98192, 5.6407, 12.1451, 33.5354, 23.7271, 17.3346 |  |
| 22H_10M | Legacy instant (cached) | 0.003 | 8101 | 128 | PASS | 0.711386 | 21.2092 | 11.064, 5.52564, 11.9819, 36.8144, 24.9766, 20.5753 |  |
| 22H_10M | Legacy instant (cached) | 0.001 | 8102 | 64 | PASS | 0.564918 | 15.9022 | 9.27704, 5.06708, 15.0139, 25.7842, 17.6125, 14.3214 |  |
| 22H_10M | Legacy instant (cached) | 0.001 | 8102 | 128 | PASS | 0.659638 | 19.0968 | 12.8091, 5.59958, 15.411, 31.8145, 20.2543, 18.2427 |  |
| 22H_10M | Legacy instant (cached) | 0.003 | 8102 | 64 | PASS | 0.614799 | 18.113 | 12.4996, 5.02288, 12.9471, 30.1682, 18.9631, 18.6994 |  |
| 22H_10M | Legacy instant (cached) | 0.003 | 8102 | 128 | PASS | 0.752305 | 22.9976 | 15.0325, 5.93785, 13.0599, 36.682, 22.822, 29.5828 |  |
| 22H_10M | Legacy instant (cached) | 0.001 | 8103 | 64 | PASS | 0.597531 | 17.2586 | 9.16554, 5.15175, 15.6425, 29.0586, 18.2852, 15.9112 |  |
| 22H_10M | Legacy instant (cached) | 0.001 | 8103 | 128 | PASS | 0.678908 | 20.0739 | 12.5631, 5.46057, 16.7657, 34.6002, 19.8929, 18.8717 |  |
| 22H_10M | Legacy instant (cached) | 0.003 | 8103 | 64 | PASS | 0.619994 | 18.0174 | 10.073, 5.60084, 13.9818, 32.0406, 19.3628, 14.7623 |  |
| 22H_10M | Legacy instant (cached) | 0.003 | 8103 | 128 | PASS | 0.706442 | 21.6337 | 12.5863, 5.62984, 14.9751, 38.8448, 20.3607, 21.6853 |  |
| 22H_10M | Causal ridge (1) | n/a | None | 64 | PASS | 0.631033 | 17.9323 | 10.2862, 6.57765, 12.4644, 27.0747, 19.8346, 22.3277 |  |
| 22H_10M | Causal ridge (1) | n/a | None | 128 | PASS | 0.811347 | 24.2445 | 14.3805, 7.80736, 14.3802, 33.8674, 24.0292, 36.4393 |  |
| 22H_10M | Causal ridge (100) | n/a | None | 64 | PASS | 0.626354 | 18.8998 | 10.1108, 5.87283, 11.7426, 29.6084, 18.4501, 25.5254 |  |
| 22H_10M | Causal ridge (100) | n/a | None | 128 | PASS | 0.740766 | 22.7888 | 13.5485, 6.48867, 12.7585, 34.3055, 21.348, 33.0898 |  |
| 22H_10M | Frozen linear AR2 | n/a | None | 64 | PASS | 0.772146 | 22.3071 | 11.1673, 7.49677, 22.2952, 34.3788, 19.9377, 26.9858 |  |
| 22H_10M | Frozen linear AR2 | n/a | None | 128 | PASS | 1.08615 | 30.8539 | 18.127, 9.07962, 37.5859, 41.8582, 25.8683, 38.2984 |  |
| 22H_10M | Persistence | n/a | None | 64 | PASS | 1.26554 | 35.9032 | 29.1171, 12.8793, 24.8447, 54.4819, 35.93, 42.9422 |  |
| 22H_10M | Persistence | n/a | None | 128 | PASS | 1.34667 | 38.3842 | 29.6839, 13.4116, 29.0819, 55.9154, 37.7355, 48.8143 |  |

## Selection

| Family | Selected rate / ridge | All options |
|---|---|---|
| Householder | 0.001 | 0.001: eligible, pooled RMSE 0.751448; 0.003: eligible, pooled RMSE 0.759592 |
| Bounded dense | 0.001 | 0.001: eligible, pooled RMSE 0.655606; 0.003: eligible, pooled RMSE 0.671532 |
| Unbounded dense | 0.003 | 0.001: eligible, pooled RMSE 0.676189; 0.003: eligible, pooled RMSE 0.672145 |
| Bounded dense + MLP | 0.001 | 0.001: eligible, pooled RMSE 0.64932; 0.003: eligible, pooled RMSE 0.654383 |
| GRU32 residual | 0.003 | 0.001: eligible, pooled RMSE 0.77244; 0.003: eligible, pooled RMSE 0.768038 |
| Legacy instant (cached) | 0.001 | 0.001: eligible, pooled RMSE 0.653591; 0.003: eligible, pooled RMSE 0.683702 |
| Causal ridge | causal_ridge_100 | [{'arm': 'causal_ridge_1', 'eligible': True, 'pooled_rmse': 0.7785094066253271}, {'arm': 'causal_ridge_100', 'eligible': True, 'pooled_rmse': 0.7088595011726029}] |

## Every fitting attempt and request cost

All30fresh attempts and6cached parent fits are retained. Legacy optimizer times are historical, not this campaign compute. Optimizer-loop duration is not inference latency; absent request timing is not zero.

| Model | Rate | Seed | Fit status | Updates | Optimizer loop (s) | Parameters | Inactive | Weight / state / buffer / normalizer (B) | Request median (ms) |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| Householder | 0.001 | 8101 | PASS | 4096 | 134.138 | 630 | 0 | 2520 / 48 / 0 / 192 | 6.43056 |
| Bounded dense | 0.001 | 8101 | PASS | 4096 | 75.7769 | 806 | 0 | 3224 / 48 / 0 / 192 | 3.98915 |
| Unbounded dense | 0.001 | 8101 | PASS | 4096 | 74.8422 | 806 | 0 | 3224 / 48 / 0 / 192 | n/a |
| Bounded dense + MLP | 0.001 | 8101 | PASS | 4096 | 60.4154 | 590 | 0 | 2360 / 48 / 0 / 192 | 3.31071 |
| GRU32 residual | 0.001 | 8101 | PASS | 4096 | 46.519 | 5916 | 0 | 23664 / 200 / 0 / 192 | n/a |
| Householder | 0.003 | 8101 | PASS | 4096 | 134.113 | 630 | 0 | 2520 / 48 / 0 / 192 | n/a |
| Bounded dense | 0.003 | 8101 | PASS | 4096 | 74.754 | 806 | 0 | 3224 / 48 / 0 / 192 | n/a |
| Unbounded dense | 0.003 | 8101 | PASS | 4096 | 74.2975 | 806 | 0 | 3224 / 48 / 0 / 192 | 3.92275 |
| Bounded dense + MLP | 0.003 | 8101 | PASS | 4096 | 60.0091 | 590 | 0 | 2360 / 48 / 0 / 192 | n/a |
| GRU32 residual | 0.003 | 8101 | PASS | 4096 | 45.9832 | 5916 | 0 | 23664 / 200 / 0 / 192 | 1.96108 |
| Householder | 0.001 | 8102 | PASS | 4096 | 134.496 | 630 | 0 | 2520 / 48 / 0 / 192 | 6.39321 |
| Bounded dense | 0.001 | 8102 | PASS | 4096 | 84.4139 | 806 | 0 | 3224 / 48 / 0 / 192 | 4.07123 |
| Unbounded dense | 0.001 | 8102 | PASS | 4096 | 83.5224 | 806 | 0 | 3224 / 48 / 0 / 192 | n/a |
| Bounded dense + MLP | 0.001 | 8102 | PASS | 4096 | 66.6671 | 590 | 0 | 2360 / 48 / 0 / 192 | 3.46537 |
| GRU32 residual | 0.001 | 8102 | PASS | 4096 | 51.2257 | 5916 | 0 | 23664 / 200 / 0 / 192 | n/a |
| Householder | 0.003 | 8102 | PASS | 4096 | 138.754 | 630 | 0 | 2520 / 48 / 0 / 192 | n/a |
| Bounded dense | 0.003 | 8102 | PASS | 4096 | 73.9294 | 806 | 0 | 3224 / 48 / 0 / 192 | n/a |
| Unbounded dense | 0.003 | 8102 | PASS | 4096 | 73.5114 | 806 | 0 | 3224 / 48 / 0 / 192 | 3.9911 |
| Bounded dense + MLP | 0.003 | 8102 | PASS | 4096 | 58.6831 | 590 | 0 | 2360 / 48 / 0 / 192 | n/a |
| GRU32 residual | 0.003 | 8102 | PASS | 4096 | 45.9489 | 5916 | 0 | 23664 / 200 / 0 / 192 | 2.03342 |
| Householder | 0.001 | 8103 | PASS | 4096 | 130.447 | 630 | 0 | 2520 / 48 / 0 / 192 | 6.41848 |
| Bounded dense | 0.001 | 8103 | PASS | 4096 | 73.9122 | 806 | 0 | 3224 / 48 / 0 / 192 | 3.89856 |
| Unbounded dense | 0.001 | 8103 | PASS | 4096 | 74.3885 | 806 | 0 | 3224 / 48 / 0 / 192 | n/a |
| Bounded dense + MLP | 0.001 | 8103 | PASS | 4096 | 61.643 | 590 | 0 | 2360 / 48 / 0 / 192 | 3.31302 |
| GRU32 residual | 0.001 | 8103 | PASS | 4096 | 46.8474 | 5916 | 0 | 23664 / 200 / 0 / 192 | n/a |
| Householder | 0.003 | 8103 | PASS | 4096 | 131.916 | 630 | 0 | 2520 / 48 / 0 / 192 | n/a |
| Bounded dense | 0.003 | 8103 | PASS | 4096 | 76.1951 | 806 | 0 | 3224 / 48 / 0 / 192 | n/a |
| Unbounded dense | 0.003 | 8103 | PASS | 4096 | 74.778 | 806 | 0 | 3224 / 48 / 0 / 192 | 3.93779 |
| Bounded dense + MLP | 0.003 | 8103 | PASS | 4096 | 60.2415 | 590 | 0 | 2360 / 48 / 0 / 192 | n/a |
| GRU32 residual | 0.003 | 8103 | PASS | 4096 | 46.1752 | 5916 | 0 | 23664 / 200 / 0 / 192 | 1.92485 |
| Legacy instant (cached) | 0.001 | 8101 | PASS | 4096 | 78.4861 | 1014 | 192 | 4056 / 48 / 0 / 192 | 3.67431 |
| Legacy instant (cached) | 0.003 | 8101 | PASS | 4096 | 77.6411 | 1014 | 192 | 4056 / 48 / 0 / 192 | n/a |
| Legacy instant (cached) | 0.001 | 8102 | PASS | 4096 | 78.2844 | 1014 | 192 | 4056 / 48 / 0 / 192 | 3.62642 |
| Legacy instant (cached) | 0.003 | 8102 | PASS | 4096 | 78.5947 | 1014 | 192 | 4056 / 48 / 0 / 192 | n/a |
| Legacy instant (cached) | 0.001 | 8103 | PASS | 4096 | 77.4723 | 1014 | 192 | 4056 / 48 / 0 / 192 | 3.65665 |
| Legacy instant (cached) | 0.003 | 8103 | PASS | 4096 | 78.0273 | 1014 | 192 | 4056 / 48 / 0 / 192 | n/a |
| Causal ridge (1) | n/a | None | reference | n/a | n/a | 445440 | 0 | 3563520 / 1536 / 16 / 192 | 0.47575 |
| Causal ridge (100) | n/a | None | reference | n/a | n/a | 445440 | 0 | 3563520 / 1536 / 16 / 192 | 0.471104 |
| Frozen linear AR2 | n/a | None | reference | n/a | n/a | 150 | 0 | 1200 / 144 / 0 / 192 | 0.229354 |
| Persistence | n/a | None | reference | n/a | n/a | 0 | 0 | 0 / 48 / 0 / 192 | 0.00856258 |

Storage is persistent numeric weights, explicit state, buffers and normalizers. Request input/output arrays are separate; temporary workspace and Python overhead are not measured. Cached legacy instant LPV stores192 inactive recurrent weights. These controls are not equal in effective capacity or numeric precision.

## All61 registered conditions

| Condition | Passed |
|---|---|
| all_selected_families_and_causal_ridge_eligible | True |
| recording_2021_12_15_21H_54M.mat/mean_within_2pct/dense_bounded | False |
| recording_2021_12_15_21H_54M.mat/seed8101_within_5pct/dense_bounded | False |
| recording_2021_12_15_21H_54M.mat/seed8102_within_5pct/dense_bounded | False |
| recording_2021_12_15_21H_54M.mat/seed8103_within_5pct/dense_bounded | False |
| recording_2021_12_15_21H_54M.mat/mean_within_2pct/dense_unbounded | False |
| recording_2021_12_15_21H_54M.mat/seed8101_within_5pct/dense_unbounded | False |
| recording_2021_12_15_21H_54M.mat/seed8102_within_5pct/dense_unbounded | False |
| recording_2021_12_15_21H_54M.mat/seed8103_within_5pct/dense_unbounded | True |
| recording_2021_12_15_21H_54M.mat/mean_within_2pct/dense_mlp | False |
| recording_2021_12_15_21H_54M.mat/seed8101_within_5pct/dense_mlp | False |
| recording_2021_12_15_21H_54M.mat/seed8102_within_5pct/dense_mlp | False |
| recording_2021_12_15_21H_54M.mat/seed8103_within_5pct/dense_mlp | False |
| recording_2021_12_15_21H_54M.mat/mean_within_2pct/gru32 | True |
| recording_2021_12_15_21H_54M.mat/seed8101_within_5pct/gru32 | False |
| recording_2021_12_15_21H_54M.mat/seed8102_within_5pct/gru32 | True |
| recording_2021_12_15_21H_54M.mat/seed8103_within_5pct/gru32 | True |
| recording_2021_12_15_21H_54M.mat/mean_within_2pct/legacy_instant | False |
| recording_2021_12_15_21H_54M.mat/seed8101_within_5pct/legacy_instant | False |
| recording_2021_12_15_21H_54M.mat/seed8102_within_5pct/legacy_instant | False |
| recording_2021_12_15_21H_54M.mat/seed8103_within_5pct/legacy_instant | False |
| recording_2021_12_15_21H_54M.mat/mean_5pct/linear_frozen | True |
| recording_2021_12_15_21H_54M.mat/mean_5pct/persistence | True |
| recording_2021_12_15_21H_54M.mat/within_5pct_causal_ridge | True |
| recording_2021_12_15_21H_54M.mat/joint0_no_10pct_harm | True |
| recording_2021_12_15_21H_54M.mat/joint1_no_10pct_harm | True |
| recording_2021_12_15_21H_54M.mat/joint2_no_10pct_harm | False |
| recording_2021_12_15_21H_54M.mat/joint3_no_10pct_harm | True |
| recording_2021_12_15_21H_54M.mat/joint4_no_10pct_harm | True |
| recording_2021_12_15_21H_54M.mat/joint5_no_10pct_harm | True |
| recording_2021_12_15_22H_10M.mat/mean_within_2pct/dense_bounded | False |
| recording_2021_12_15_22H_10M.mat/seed8101_within_5pct/dense_bounded | False |
| recording_2021_12_15_22H_10M.mat/seed8102_within_5pct/dense_bounded | False |
| recording_2021_12_15_22H_10M.mat/seed8103_within_5pct/dense_bounded | False |
| recording_2021_12_15_22H_10M.mat/mean_within_2pct/dense_unbounded | False |
| recording_2021_12_15_22H_10M.mat/seed8101_within_5pct/dense_unbounded | False |
| recording_2021_12_15_22H_10M.mat/seed8102_within_5pct/dense_unbounded | False |
| recording_2021_12_15_22H_10M.mat/seed8103_within_5pct/dense_unbounded | False |
| recording_2021_12_15_22H_10M.mat/mean_within_2pct/dense_mlp | False |
| recording_2021_12_15_22H_10M.mat/seed8101_within_5pct/dense_mlp | False |
| recording_2021_12_15_22H_10M.mat/seed8102_within_5pct/dense_mlp | False |
| recording_2021_12_15_22H_10M.mat/seed8103_within_5pct/dense_mlp | False |
| recording_2021_12_15_22H_10M.mat/mean_within_2pct/gru32 | True |
| recording_2021_12_15_22H_10M.mat/seed8101_within_5pct/gru32 | True |
| recording_2021_12_15_22H_10M.mat/seed8102_within_5pct/gru32 | True |
| recording_2021_12_15_22H_10M.mat/seed8103_within_5pct/gru32 | False |
| recording_2021_12_15_22H_10M.mat/mean_within_2pct/legacy_instant | False |
| recording_2021_12_15_22H_10M.mat/seed8101_within_5pct/legacy_instant | False |
| recording_2021_12_15_22H_10M.mat/seed8102_within_5pct/legacy_instant | False |
| recording_2021_12_15_22H_10M.mat/seed8103_within_5pct/legacy_instant | False |
| recording_2021_12_15_22H_10M.mat/mean_5pct/linear_frozen | True |
| recording_2021_12_15_22H_10M.mat/mean_5pct/persistence | True |
| recording_2021_12_15_22H_10M.mat/within_5pct_causal_ridge | False |
| recording_2021_12_15_22H_10M.mat/joint0_no_10pct_harm | False |
| recording_2021_12_15_22H_10M.mat/joint1_no_10pct_harm | True |
| recording_2021_12_15_22H_10M.mat/joint2_no_10pct_harm | False |
| recording_2021_12_15_22H_10M.mat/joint3_no_10pct_harm | True |
| recording_2021_12_15_22H_10M.mat/joint4_no_10pct_harm | True |
| recording_2021_12_15_22H_10M.mat/joint5_no_10pct_harm | True |
| at_most_80pct_dense_bounded_latency | False |
| at_most_80pct_dense_bounded_numeric_storage | True |

The forced-state bound for the bounded variants applies with finite fixed weights and bounded inputs in exact arithmetic. It is not incremental contraction, a bounded-gradient guarantee, or a robot-safety result. A pass only qualifies a separately designed confirmation protocol; the current recordings remain development data.
