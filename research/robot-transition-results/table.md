# Bounded robot transition: complete saved results

Development on the same two previously exposed recordings. Both rates are retained in the complete table. Selected-rate dots show all three individual fits; accuracy diamonds are arithmetic means, not an ensemble. 32 observed steps followed by 128 forecast steps (12.8 s), conditional on realized measured torque. Each forecast uses only preceding torque inputs. Internal CONFIRM and official TEST remain closed. Different model sizes and precision; no novelty, control-safety or official benchmark claim.

Saved outcome: **DO_NOT_ADVANCE_TRANSITION**, 27/45 conditions.

## Every candidate and reference

All 112 rows are retained. H64 is descriptive; two-rate selection and the rule use pooled DEV H128. Both causal ridge penalties are visible.

| Recording | Model | Rate | Seed | Horizon | Status | Standardized RMSE | Physical RMSE (deg) | Per-joint RMSE (deg) | Error |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| 21H_54M | Recurrent LPV | 0.001 | 8101 | 64 | PASS | 0.546451 | 16.0727 | 10.5893, 3.81235, 13.0142, 28.318, 17.5015, 12.0722 |  |
| 21H_54M | Recurrent LPV | 0.001 | 8101 | 128 | PASS | 0.648312 | 19.972 | 13.7372, 4.00157, 13.2681, 37.849, 19.5279, 14.0939 |  |
| 21H_54M | Instant LPV | 0.001 | 8101 | 64 | PASS | 0.56324 | 16.9653 | 6.67805, 4.02551, 13.9768, 32.0966, 18.0975, 10.6332 |  |
| 21H_54M | Instant LPV | 0.001 | 8101 | 128 | PASS | 0.65997 | 20.4468 | 8.94, 4.42939, 14.7836, 39.6818, 20.6215, 13.7996 |  |
| 21H_54M | Constant LPV | 0.001 | 8101 | 64 | PASS | 0.617192 | 18.6637 | 11.0762, 4.60357, 13.6342, 30.523, 18.8895, 21.72 |  |
| 21H_54M | Constant LPV | 0.001 | 8101 | 128 | PASS | 0.697016 | 21.2074 | 12.7607, 5.18237, 14.1749, 35.1411, 21.5793, 24.6442 |  |
| 21H_54M | GRU residual | 0.001 | 8101 | 64 | PASS | 0.641013 | 18.319 | 12.3115, 4.81138, 12.606, 26.7989, 23.0884, 20.7032 |  |
| 21H_54M | GRU residual | 0.001 | 8101 | 128 | PASS | 0.734679 | 21.6851 | 14.0701, 4.97328, 13.638, 32.9843, 25.9558, 25.5165 |  |
| 21H_54M | Recurrent LPV | 0.003 | 8101 | 64 | PASS | 0.557771 | 16.11 | 8.74675, 3.82015, 12.7403, 27.9247, 20.0865, 10.978 |  |
| 21H_54M | Recurrent LPV | 0.003 | 8101 | 128 | PASS | 0.63828 | 18.6184 | 10.0161, 4.15134, 13.4057, 32.4182, 23.4851, 13.4202 |  |
| 21H_54M | Instant LPV | 0.003 | 8101 | 64 | PASS | 0.575331 | 16.8546 | 8.24717, 3.76089, 12.069, 29.5873, 21.2465, 12.2405 |  |
| 21H_54M | Instant LPV | 0.003 | 8101 | 128 | PASS | 0.642299 | 18.876 | 9.22483, 4.28916, 11.9004, 32.4879, 24.2005, 15.8613 |  |
| 21H_54M | Constant LPV | 0.003 | 8101 | 64 | PASS | 0.639179 | 19.0844 | 12.4121, 4.75709, 13.5532, 31.1381, 20.1859, 21.1628 |  |
| 21H_54M | Constant LPV | 0.003 | 8101 | 128 | PASS | 0.708562 | 21.4876 | 13.1374, 5.25824, 13.8661, 35.9891, 22.3945, 24.1054 |  |
| 21H_54M | GRU residual | 0.003 | 8101 | 64 | PASS | 0.591319 | 16.8812 | 10.4742, 4.59591, 12.3921, 25.4442, 21.0726, 18.2755 |  |
| 21H_54M | GRU residual | 0.003 | 8101 | 128 | PASS | 0.676294 | 19.7106 | 11.9841, 5.12686, 14.273, 30.3032, 23.2093, 22.3711 |  |
| 21H_54M | Recurrent LPV | 0.001 | 8102 | 64 | PASS | 0.568157 | 16.455 | 8.60076, 4.75791, 13.0123, 26.755, 18.9247, 16.8732 |  |
| 21H_54M | Recurrent LPV | 0.001 | 8102 | 128 | PASS | 0.649434 | 19.3461 | 12.1121, 4.769, 13.2111, 32.6902, 21.3295, 19.4437 |  |
| 21H_54M | Instant LPV | 0.001 | 8102 | 64 | PASS | 0.514277 | 14.5716 | 8.30987, 4.05307, 11.6938, 22.5592, 18.3277, 14.385 |  |
| 21H_54M | Instant LPV | 0.001 | 8102 | 128 | PASS | 0.605261 | 17.864 | 10.8638, 4.42171, 12.2224, 29.6511, 20.6557, 17.9421 |  |
| 21H_54M | Constant LPV | 0.001 | 8102 | 64 | PASS | 0.588396 | 17.8075 | 9.10854, 5.06388, 12.4095, 29.5483, 17.59, 21.3901 |  |
| 21H_54M | Constant LPV | 0.001 | 8102 | 128 | PASS | 0.667558 | 20.1045 | 11.6906, 5.6112, 12.8563, 33.1194, 20.6556, 23.8361 |  |
| 21H_54M | GRU residual | 0.001 | 8102 | 64 | PASS | 0.614892 | 17.5913 | 11.2442, 5.2751, 11.5828, 26.5795, 21.3839, 20.1139 |  |
| 21H_54M | GRU residual | 0.001 | 8102 | 128 | PASS | 0.67277 | 19.7925 | 13.9033, 5.30497, 12.1053, 29.7834, 22.2988, 24.4579 |  |
| 21H_54M | Recurrent LPV | 0.003 | 8102 | 64 | PASS | 0.618545 | 18.7639 | 9.57632, 3.96957, 14.7359, 35.989, 19.807, 10.0182 |  |
| 21H_54M | Recurrent LPV | 0.003 | 8102 | 128 | PASS | 0.713024 | 21.6727 | 10.4997, 4.26619, 15.8494, 40.2782, 24.1526, 15.2612 |  |
| 21H_54M | Instant LPV | 0.003 | 8102 | 64 | PASS | 0.564161 | 17.1923 | 12.3704, 3.31926, 11.3339, 30.7248, 17.7581, 14.8855 |  |
| 21H_54M | Instant LPV | 0.003 | 8102 | 128 | PASS | 0.632774 | 19.2623 | 13.3839, 3.7759, 11.6092, 34.5008, 20.7915, 16.5972 |  |
| 21H_54M | Constant LPV | 0.003 | 8102 | 64 | PASS | 0.628386 | 18.927 | 12.7207, 5.05709, 12.4706, 31.7211, 18.752, 21.1804 |  |
| 21H_54M | Constant LPV | 0.003 | 8102 | 128 | PASS | 0.675196 | 20.4164 | 12.9634, 5.52383, 12.7456, 34.3687, 20.4885, 23.216 |  |
| 21H_54M | GRU residual | 0.003 | 8102 | 64 | PASS | 0.619471 | 18.094 | 13.6042, 4.70676, 13.0443, 22.788, 19.8207, 25.9773 |  |
| 21H_54M | GRU residual | 0.003 | 8102 | 128 | PASS | 0.708721 | 21.4036 | 15.3595, 4.82542, 12.8705, 28.8846, 22.8706, 31.0876 |  |
| 21H_54M | Recurrent LPV | 0.001 | 8103 | 64 | PASS | 0.46505 | 13.1741 | 7.0938, 3.35273, 11.6993, 21.7683, 16.5767, 9.70917 |  |
| 21H_54M | Recurrent LPV | 0.001 | 8103 | 128 | PASS | 0.58774 | 17.2664 | 9.40169, 3.54986, 12.2476, 30.8768, 21.6604, 10.7345 |  |
| 21H_54M | Instant LPV | 0.001 | 8103 | 64 | PASS | 0.515074 | 15.1888 | 6.42573, 3.48004, 13.2965, 25.8463, 17.4322, 13.4936 |  |
| 21H_54M | Instant LPV | 0.001 | 8103 | 128 | PASS | 0.612232 | 18.2915 | 8.12166, 3.78753, 14.1794, 31.9979, 21.4707, 15.5325 |  |
| 21H_54M | Constant LPV | 0.001 | 8103 | 64 | PASS | 0.596466 | 18.0126 | 9.40611, 4.78334, 13.7566, 28.5234, 17.9106, 22.6221 |  |
| 21H_54M | Constant LPV | 0.001 | 8103 | 128 | PASS | 0.686013 | 20.6478 | 11.771, 5.51036, 14.5869, 33.0949, 21.1437, 25.1785 |  |
| 21H_54M | GRU residual | 0.001 | 8103 | 64 | PASS | 0.626801 | 18.2039 | 13.7852, 5.33803, 12.9299, 29.2446, 19.1945, 19.4658 |  |
| 21H_54M | GRU residual | 0.001 | 8103 | 128 | PASS | 0.680152 | 20.1444 | 15.2674, 5.54509, 12.5315, 32.8518, 20.8418, 22.3672 |  |
| 21H_54M | Recurrent LPV | 0.003 | 8103 | 64 | PASS | 0.599121 | 17.8901 | 9.94107, 3.40674, 12.3114, 32.8358, 21.5373, 10.7841 |  |
| 21H_54M | Recurrent LPV | 0.003 | 8103 | 128 | PASS | 0.696586 | 21.7416 | 10.3982, 3.8364, 12.3698, 42.4771, 23.9649, 13.4798 |  |
| 21H_54M | Instant LPV | 0.003 | 8103 | 64 | PASS | 0.571113 | 16.7186 | 8.92731, 3.74431, 13.919, 28.3508, 19.7328, 14.0166 |  |
| 21H_54M | Instant LPV | 0.003 | 8103 | 128 | PASS | 0.648573 | 19.4742 | 10.1487, 4.22734, 14.2676, 33.9105, 22.0256, 17.776 |  |
| 21H_54M | Constant LPV | 0.003 | 8103 | 64 | PASS | 0.626062 | 19.0159 | 10.242, 4.83661, 13.3217, 30.5615, 19.3938, 23.5315 |  |
| 21H_54M | Constant LPV | 0.003 | 8103 | 128 | PASS | 0.705156 | 21.408 | 12.2662, 5.43282, 13.8926, 35.0107, 22.2012, 25.6553 |  |
| 21H_54M | GRU residual | 0.003 | 8103 | 64 | PASS | 0.576045 | 17.3416 | 11.1701, 4.41162, 11.5495, 27.0474, 17.9461, 21.752 |  |
| 21H_54M | GRU residual | 0.003 | 8103 | 128 | PASS | 0.661731 | 20.1396 | 12.3534, 4.59439, 11.6325, 31.3899, 21.9584, 25.6339 |  |
| 21H_54M | Causal ridge (1) | n/a | None | 64 | PASS | 0.559915 | 16.2719 | 10.4995, 5.65965, 12.582, 27.7244, 15.1837, 16.9964 |  |
| 21H_54M | Causal ridge (1) | n/a | None | 128 | PASS | 0.744225 | 21.9132 | 14.0532, 7.23159, 15.9085, 36.3675, 20.6251, 25.1049 |  |
| 21H_54M | Causal ridge (100) | n/a | None | 64 | PASS | 0.584402 | 17.3852 | 9.47897, 4.95741, 12.2164, 28.6224, 18.4429, 19.7593 |  |
| 21H_54M | Causal ridge (100) | n/a | None | 128 | PASS | 0.675447 | 20.2353 | 11.7595, 5.7122, 13.5981, 33.4459, 20.8447, 23.4061 |  |
| 21H_54M | Frozen linear AR2 | n/a | None | 64 | PASS | 0.730026 | 21.2211 | 10.7519, 5.33914, 23.0623, 28.0936, 20.8668, 28.3084 |  |
| 21H_54M | Frozen linear AR2 | n/a | None | 128 | PASS | 1.08777 | 30.8077 | 19.7813, 7.27011, 40.4915, 37.0243, 26.0793, 39.4976 |  |
| 21H_54M | Persistence | n/a | None | 64 | PASS | 1.13736 | 32.6754 | 20.8189, 10.4477, 29.6569, 49.0193, 32.2645, 39.2438 |  |
| 21H_54M | Persistence | n/a | None | 128 | PASS | 1.23294 | 35.4162 | 26.9324, 11.1036, 31.1057, 53.3282, 33.6346, 41.6469 |  |
| 22H_10M | Recurrent LPV | 0.001 | 8101 | 64 | PASS | 0.580122 | 16.8556 | 8.73295, 5.43099, 14.2625, 28.4412, 17.1817, 17.0699 |  |
| 22H_10M | Recurrent LPV | 0.001 | 8101 | 128 | PASS | 0.700807 | 20.634 | 12.3985, 6.26691, 15.3059, 33.8641, 21.1114, 23.1267 |  |
| 22H_10M | Instant LPV | 0.001 | 8101 | 64 | PASS | 0.637721 | 18.8098 | 9.04651, 5.18593, 13.5674, 32.0806, 21.4383, 18.474 |  |
| 22H_10M | Instant LPV | 0.001 | 8101 | 128 | PASS | 0.700229 | 20.8666 | 12.3298, 5.32696, 14.0171, 33.5766, 23.1534, 23.9194 |  |
| 22H_10M | Constant LPV | 0.001 | 8101 | 64 | PASS | 0.685828 | 20.9931 | 12.2038, 6.85828, 13.0011, 33.6961, 17.163, 29.1422 |  |
| 22H_10M | Constant LPV | 0.001 | 8101 | 128 | PASS | 0.751391 | 23.1604 | 14.6554, 7.00783, 13.8578, 35.9494, 19.2017, 33.1877 |  |
| 22H_10M | GRU residual | 0.001 | 8101 | 64 | PASS | 0.693956 | 20.9204 | 15.3887, 5.87227, 13.4029, 36.338, 19.2908, 21.965 |  |
| 22H_10M | GRU residual | 0.001 | 8101 | 128 | PASS | 0.764716 | 23.3399 | 18.1029, 6.09212, 14.1521, 40.9016, 20.6943, 24.5396 |  |
| 22H_10M | Recurrent LPV | 0.003 | 8101 | 64 | PASS | 0.565406 | 16.6464 | 8.78613, 4.53111, 10.9851, 29.0387, 19.4741, 14.8908 |  |
| 22H_10M | Recurrent LPV | 0.003 | 8101 | 128 | PASS | 0.675948 | 20.3544 | 11.0805, 4.65701, 12.3363, 34.4001, 23.5484, 21.2434 |  |
| 22H_10M | Instant LPV | 0.003 | 8101 | 64 | PASS | 0.668639 | 19.4381 | 9.98192, 5.6407, 12.1451, 33.5354, 23.7271, 17.3346 |  |
| 22H_10M | Instant LPV | 0.003 | 8101 | 128 | PASS | 0.711386 | 21.2092 | 11.064, 5.52564, 11.9819, 36.8144, 24.9766, 20.5753 |  |
| 22H_10M | Constant LPV | 0.003 | 8101 | 64 | PASS | 0.69072 | 20.7662 | 13.3947, 6.92117, 13.5335, 33.4317, 17.6442, 27.3484 |  |
| 22H_10M | Constant LPV | 0.003 | 8101 | 128 | PASS | 0.753285 | 22.8727 | 15.4378, 7.05567, 13.8338, 35.417, 19.981, 31.7154 |  |
| 22H_10M | GRU residual | 0.003 | 8101 | 64 | PASS | 0.634859 | 18.4663 | 8.59047, 5.50771, 11.6194, 28.4283, 22.5409, 22.1498 |  |
| 22H_10M | GRU residual | 0.003 | 8101 | 128 | PASS | 0.69622 | 20.5479 | 10.9902, 5.53984, 12.6379, 31.2676, 24.3686, 25.5072 |  |
| 22H_10M | Recurrent LPV | 0.001 | 8102 | 64 | PASS | 0.686931 | 20.8477 | 9.61668, 6.06121, 13.9142, 36.3322, 20.7614, 23.1055 |  |
| 22H_10M | Recurrent LPV | 0.001 | 8102 | 128 | PASS | 0.81629 | 25.395 | 13.3039, 6.28213, 15.0841, 40.6073, 24.7912, 34.0864 |  |
| 22H_10M | Instant LPV | 0.001 | 8102 | 64 | PASS | 0.564918 | 15.9022 | 9.27704, 5.06708, 15.0139, 25.7842, 17.6125, 14.3214 |  |
| 22H_10M | Instant LPV | 0.001 | 8102 | 128 | PASS | 0.659638 | 19.0968 | 12.8091, 5.59958, 15.411, 31.8145, 20.2543, 18.2427 |  |
| 22H_10M | Constant LPV | 0.001 | 8102 | 64 | PASS | 0.705502 | 21.9081 | 11.2664, 6.64444, 14.8931, 36.9694, 17.3524, 28.6191 |  |
| 22H_10M | Constant LPV | 0.001 | 8102 | 128 | PASS | 0.785751 | 24.837 | 14.2677, 6.83964, 14.9188, 40.1118, 19.4521, 35.2278 |  |
| 22H_10M | GRU residual | 0.001 | 8102 | 64 | PASS | 0.69805 | 20.313 | 14.8183, 5.66945, 13.5877, 31.1993, 22.6734, 23.4921 |  |
| 22H_10M | GRU residual | 0.001 | 8102 | 128 | PASS | 0.747626 | 22.1332 | 16.5845, 5.7871, 13.4253, 34.4792, 23.917, 26.2614 |  |
| 22H_10M | Recurrent LPV | 0.003 | 8102 | 64 | PASS | 0.747827 | 22.5934 | 10.6078, 6.10725, 16.0173, 41.1031, 23.3979, 20.4809 |  |
| 22H_10M | Recurrent LPV | 0.003 | 8102 | 128 | PASS | 0.789072 | 24.1456 | 13.3948, 6.35332, 15.5005, 41.1369, 23.8317, 27.8891 |  |
| 22H_10M | Instant LPV | 0.003 | 8102 | 64 | PASS | 0.614799 | 18.113 | 12.4996, 5.02288, 12.9471, 30.1682, 18.9631, 18.6994 |  |
| 22H_10M | Instant LPV | 0.003 | 8102 | 128 | PASS | 0.752305 | 22.9976 | 15.0325, 5.93785, 13.0599, 36.682, 22.822, 29.5828 |  |
| 22H_10M | Constant LPV | 0.003 | 8102 | 64 | PASS | 0.72198 | 22.3004 | 14.7825, 7.16488, 12.9551, 36.8762, 16.5326, 30.2154 |  |
| 22H_10M | Constant LPV | 0.003 | 8102 | 128 | PASS | 0.781268 | 24.5 | 16.5489, 7.14114, 13.2485, 39.4621, 18.4778, 34.6759 |  |
| 22H_10M | GRU residual | 0.003 | 8102 | 64 | PASS | 0.717901 | 21.5222 | 13.8877, 5.89557, 11.5667, 32.3967, 23.324, 28.7098 |  |
| 22H_10M | GRU residual | 0.003 | 8102 | 128 | PASS | 0.762423 | 23.0794 | 15.3874, 6.04413, 11.3173, 34.8446, 24.6223, 31.2116 |  |
| 22H_10M | Recurrent LPV | 0.001 | 8103 | 64 | PASS | 0.617582 | 18.8061 | 8.59881, 4.9614, 13.6057, 34.0532, 18.8652, 17.9672 |  |
| 22H_10M | Recurrent LPV | 0.001 | 8103 | 128 | PASS | 0.736526 | 22.6974 | 12.7083, 5.38407, 14.9738, 38.8864, 22.4459, 25.6974 |  |
| 22H_10M | Instant LPV | 0.001 | 8103 | 64 | PASS | 0.597531 | 17.2586 | 9.16554, 5.15175, 15.6425, 29.0586, 18.2852, 15.9112 |  |
| 22H_10M | Instant LPV | 0.001 | 8103 | 128 | PASS | 0.678908 | 20.0739 | 12.5631, 5.46057, 16.7657, 34.6002, 19.8929, 18.8717 |  |
| 22H_10M | Constant LPV | 0.001 | 8103 | 64 | PASS | 0.658681 | 19.993 | 11.1757, 6.79797, 13.622, 33.6354, 15.9134, 25.6335 |  |
| 22H_10M | Constant LPV | 0.001 | 8103 | 128 | PASS | 0.733252 | 22.723 | 14.0435, 6.8477, 14.1313, 37.0678, 17.9496, 30.9513 |  |
| 22H_10M | GRU residual | 0.001 | 8103 | 64 | PASS | 0.675731 | 19.7209 | 15.051, 5.5439, 12.752, 30.7682, 21.4478, 22.5144 |  |
| 22H_10M | GRU residual | 0.001 | 8103 | 128 | PASS | 0.729056 | 21.3656 | 17.4341, 6.07854, 12.1089, 33.1873, 22.7317, 25.1649 |  |
| 22H_10M | Recurrent LPV | 0.003 | 8103 | 64 | PASS | 0.618887 | 18.4866 | 10.1009, 5.12237, 14.0821, 29.9071, 18.832, 21.7915 |  |
| 22H_10M | Recurrent LPV | 0.003 | 8103 | 128 | PASS | 0.738792 | 22.8196 | 11.7097, 5.50799, 17.0397, 32.2222, 21.4532, 34.1771 |  |
| 22H_10M | Instant LPV | 0.003 | 8103 | 64 | PASS | 0.619994 | 18.0174 | 10.073, 5.60084, 13.9818, 32.0406, 19.3628, 14.7623 |  |
| 22H_10M | Instant LPV | 0.003 | 8103 | 128 | PASS | 0.706442 | 21.6337 | 12.5863, 5.62984, 14.9751, 38.8448, 20.3607, 21.6853 |  |
| 22H_10M | Constant LPV | 0.003 | 8103 | 64 | PASS | 0.677044 | 20.6251 | 11.3023, 6.77721, 13.6576, 33.9286, 17.159, 27.3237 |  |
| 22H_10M | Constant LPV | 0.003 | 8103 | 128 | PASS | 0.748268 | 23.3579 | 14.0655, 6.79282, 13.8541, 38.1834, 18.8018, 32.0336 |  |
| 22H_10M | GRU residual | 0.003 | 8103 | 64 | PASS | 0.68935 | 20.616 | 14.67, 5.98322, 13.7912, 34.2288, 19.5154, 23.5893 |  |
| 22H_10M | GRU residual | 0.003 | 8103 | 128 | PASS | 0.718637 | 21.6168 | 16.4782, 5.9198, 14.0482, 34.1689, 20.1146, 26.9754 |  |
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
| Recurrent LPV | 0.001 | 0.001: eligible, pooled RMSE 0.693715; 0.003: eligible, pooled RMSE 0.71021 |
| Instant LPV | 0.001 | 0.001: eligible, pooled RMSE 0.653591; 0.003: eligible, pooled RMSE 0.683702 |
| Constant LPV | 0.001 | 0.001: eligible, pooled RMSE 0.72131; 0.003: eligible, pooled RMSE 0.729488 |
| GRU residual | 0.003 | 0.001: eligible, pooled RMSE 0.722292; 0.003: eligible, pooled RMSE 0.704745 |
| Causal ridge | causal_ridge_100 | [{'arm': 'causal_ridge_1', 'eligible': True, 'pooled_rmse': 0.7785094066253271}, {'arm': 'causal_ridge_100', 'eligible': True, 'pooled_rmse': 0.7088595011726029}] |

## Every fitting attempt and request cost

All 24 attempts are retained. Optimizer-loop duration is not inference latency; absent request timing is not zero.

| Model | Rate | Seed | Fit status | Updates | Optimizer loop (s) | Parameters | Inactive | Weight / state / buffer / normalizer (B) | Request median (ms) |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|
| Recurrent LPV | 0.001 | 8101 | PASS | 4096 | 83.6374 | 1014 | 0 | 4056 / 80 / 0 / 192 | 4.04471 |
| Instant LPV | 0.001 | 8101 | PASS | 4096 | 78.4861 | 1014 | 192 | 4056 / 48 / 0 / 192 | 3.71175 |
| Constant LPV | 0.001 | 8101 | PASS | 4096 | 40.9498 | 468 | 0 | 1872 / 48 / 0 / 192 | 2.38531 |
| GRU residual | 0.001 | 8101 | PASS | 4096 | 46.2055 | 1296 | 0 | 5184 / 112 / 0 / 192 | n/a |
| Recurrent LPV | 0.003 | 8101 | PASS | 4096 | 84.5898 | 1014 | 0 | 4056 / 80 / 0 / 192 | n/a |
| Instant LPV | 0.003 | 8101 | PASS | 4096 | 77.6411 | 1014 | 192 | 4056 / 48 / 0 / 192 | n/a |
| Constant LPV | 0.003 | 8101 | PASS | 4096 | 40.9772 | 468 | 0 | 1872 / 48 / 0 / 192 | n/a |
| GRU residual | 0.003 | 8101 | PASS | 4096 | 45.7807 | 1296 | 0 | 5184 / 112 / 0 / 192 | 2.02752 |
| Recurrent LPV | 0.001 | 8102 | PASS | 4096 | 84.1329 | 1014 | 0 | 4056 / 80 / 0 / 192 | 4.17969 |
| Instant LPV | 0.001 | 8102 | PASS | 4096 | 78.2844 | 1014 | 192 | 4056 / 48 / 0 / 192 | 3.84542 |
| Constant LPV | 0.001 | 8102 | PASS | 4096 | 40.8682 | 468 | 0 | 1872 / 48 / 0 / 192 | 2.3419 |
| GRU residual | 0.001 | 8102 | PASS | 4096 | 46.2541 | 1296 | 0 | 5184 / 112 / 0 / 192 | n/a |
| Recurrent LPV | 0.003 | 8102 | PASS | 4096 | 83.7658 | 1014 | 0 | 4056 / 80 / 0 / 192 | n/a |
| Instant LPV | 0.003 | 8102 | PASS | 4096 | 78.5947 | 1014 | 192 | 4056 / 48 / 0 / 192 | n/a |
| Constant LPV | 0.003 | 8102 | PASS | 4096 | 40.8417 | 468 | 0 | 1872 / 48 / 0 / 192 | n/a |
| GRU residual | 0.003 | 8102 | PASS | 4096 | 46.1497 | 1296 | 0 | 5184 / 112 / 0 / 192 | 1.96933 |
| Recurrent LPV | 0.001 | 8103 | PASS | 4096 | 83.8572 | 1014 | 0 | 4056 / 80 / 0 / 192 | 4.10627 |
| Instant LPV | 0.001 | 8103 | PASS | 4096 | 77.4723 | 1014 | 192 | 4056 / 48 / 0 / 192 | 3.92898 |
| Constant LPV | 0.001 | 8103 | PASS | 4096 | 41.645 | 468 | 0 | 1872 / 48 / 0 / 192 | 2.39598 |
| GRU residual | 0.001 | 8103 | PASS | 4096 | 46.0642 | 1296 | 0 | 5184 / 112 / 0 / 192 | n/a |
| Recurrent LPV | 0.003 | 8103 | PASS | 4096 | 84.0674 | 1014 | 0 | 4056 / 80 / 0 / 192 | n/a |
| Instant LPV | 0.003 | 8103 | PASS | 4096 | 78.0273 | 1014 | 192 | 4056 / 48 / 0 / 192 | n/a |
| Constant LPV | 0.003 | 8103 | PASS | 4096 | 42.5415 | 468 | 0 | 1872 / 48 / 0 / 192 | n/a |
| GRU residual | 0.003 | 8103 | PASS | 4096 | 49.4194 | 1296 | 0 | 5184 / 112 / 0 / 192 | 2.04454 |
| Causal ridge (1) | n/a | None | reference | n/a | n/a | 445440 | 0 | 3563520 / 1536 / 16 / 192 | 0.517729 |
| Causal ridge (100) | n/a | None | reference | n/a | n/a | 445440 | 0 | 3563520 / 1536 / 16 / 192 | 0.517958 |
| Frozen linear AR2 | n/a | None | reference | n/a | n/a | 150 | 0 | 1200 / 144 / 0 / 192 | 0.247417 |
| Persistence | n/a | None | reference | n/a | n/a | 0 | 0 | 0 / 48 / 0 / 192 | 0.00900042 |

Storage is persistent numeric weights, explicit state, buffers and normalizers. Request input/output arrays are separate; temporary workspace and Python overhead are not measured. Instant LPV stores 192 inactive recurrent weights. These controls are not equal in effective capacity or numeric precision.

## All 45 registered conditions

| Condition | Passed |
|---|---|
| all_selected_families_and_causal_ridge_eligible | True |
| recording_2021_12_15_21H_54M.mat/mean_5pct/lpv_instant | False |
| recording_2021_12_15_21H_54M.mat/seed8101_2pct/lpv_instant | False |
| recording_2021_12_15_21H_54M.mat/seed8102_2pct/lpv_instant | False |
| recording_2021_12_15_21H_54M.mat/seed8103_2pct/lpv_instant | True |
| recording_2021_12_15_21H_54M.mat/mean_5pct/lpv_constant | True |
| recording_2021_12_15_21H_54M.mat/seed8101_2pct/lpv_constant | True |
| recording_2021_12_15_21H_54M.mat/seed8102_2pct/lpv_constant | True |
| recording_2021_12_15_21H_54M.mat/seed8103_2pct/lpv_constant | True |
| recording_2021_12_15_21H_54M.mat/mean_5pct/gru_residual | True |
| recording_2021_12_15_21H_54M.mat/seed8101_2pct/gru_residual | True |
| recording_2021_12_15_21H_54M.mat/seed8102_2pct/gru_residual | True |
| recording_2021_12_15_21H_54M.mat/seed8103_2pct/gru_residual | True |
| recording_2021_12_15_21H_54M.mat/mean_5pct/linear_frozen | True |
| recording_2021_12_15_21H_54M.mat/mean_5pct/persistence | True |
| recording_2021_12_15_21H_54M.mat/within_5pct_causal_ridge | True |
| recording_2021_12_15_21H_54M.mat/joint0_no_10pct_harm | True |
| recording_2021_12_15_21H_54M.mat/joint1_no_10pct_harm | True |
| recording_2021_12_15_21H_54M.mat/joint2_no_10pct_harm | True |
| recording_2021_12_15_21H_54M.mat/joint3_no_10pct_harm | False |
| recording_2021_12_15_21H_54M.mat/joint4_no_10pct_harm | True |
| recording_2021_12_15_21H_54M.mat/joint5_no_10pct_harm | True |
| recording_2021_12_15_22H_10M.mat/mean_5pct/lpv_instant | False |
| recording_2021_12_15_22H_10M.mat/seed8101_2pct/lpv_instant | False |
| recording_2021_12_15_22H_10M.mat/seed8102_2pct/lpv_instant | False |
| recording_2021_12_15_22H_10M.mat/seed8103_2pct/lpv_instant | False |
| recording_2021_12_15_22H_10M.mat/mean_5pct/lpv_constant | False |
| recording_2021_12_15_22H_10M.mat/seed8101_2pct/lpv_constant | True |
| recording_2021_12_15_22H_10M.mat/seed8102_2pct/lpv_constant | False |
| recording_2021_12_15_22H_10M.mat/seed8103_2pct/lpv_constant | False |
| recording_2021_12_15_22H_10M.mat/mean_5pct/gru_residual | False |
| recording_2021_12_15_22H_10M.mat/seed8101_2pct/gru_residual | False |
| recording_2021_12_15_22H_10M.mat/seed8102_2pct/gru_residual | False |
| recording_2021_12_15_22H_10M.mat/seed8103_2pct/gru_residual | False |
| recording_2021_12_15_22H_10M.mat/mean_5pct/linear_frozen | True |
| recording_2021_12_15_22H_10M.mat/mean_5pct/persistence | True |
| recording_2021_12_15_22H_10M.mat/within_5pct_causal_ridge | True |
| recording_2021_12_15_22H_10M.mat/joint0_no_10pct_harm | True |
| recording_2021_12_15_22H_10M.mat/joint1_no_10pct_harm | True |
| recording_2021_12_15_22H_10M.mat/joint2_no_10pct_harm | False |
| recording_2021_12_15_22H_10M.mat/joint3_no_10pct_harm | False |
| recording_2021_12_15_22H_10M.mat/joint4_no_10pct_harm | True |
| recording_2021_12_15_22H_10M.mat/joint5_no_10pct_harm | True |
| at_most_twice_gru_latency | False |
| at_most_gru_numeric_storage | True |

The forced-state bound for LPV applies with finite fixed weights and bounded inputs in exact arithmetic. It is not incremental contraction, a bounded-gradient guarantee, or a robot-safety result. A pass only qualifies a separately designed confirmation protocol; the current recordings remain development data.
