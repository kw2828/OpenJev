# Preserved collection quota failure

The fixed 72-trajectory collection finished 52,072 steps, then failed its required row quotas. 3 of 18 stratum/collector cells were short. No mixture, training or evaluation was admitted.

| Setting | Initial hit | Collector | Available | Required | Shortfall |
|---|---:|---:|---:|---:|---:|
| lambda3 | 1 | 10101 | 4400 | 246 | 0 |
| lambda3 | 1 | 10102 | 4399 | 245 | 0 |
| lambda3 | 1 | 10103 | 4399 | 245 | 0 |
| lambda3 | 2 | 10101 | 35 | 122 | 87 |
| lambda3 | 2 | 10102 | 23 | 121 | 98 |
| lambda3 | 2 | 10103 | 31 | 121 | 90 |
| lambda3 | 3 | 10101 | 2199 | 65 | 0 |
| lambda3 | 3 | 10102 | 2199 | 65 | 0 |
| lambda3 | 3 | 10103 | 2349 | 64 | 0 |
| lambda4 | 1 | 10101 | 6601 | 259 | 0 |
| lambda4 | 1 | 10102 | 6603 | 259 | 0 |
| lambda4 | 1 | 10103 | 6617 | 259 | 0 |
| lambda4 | 2 | 10101 | 4409 | 157 | 0 |
| lambda4 | 2 | 10102 | 2227 | 157 | 0 |
| lambda4 | 2 | 10103 | 2253 | 156 | 0 |
| lambda4 | 3 | 10101 | 2462 | 83 | 0 |
| lambda4 | 3 | 10102 | 669 | 83 | 0 |
| lambda4 | 3 | 10103 | 197 | 83 | 0 |

All 72 episode identities follow the frozen rotation, with four episodes per cell. The count for each episode is its number of pre-action states, indices 0 through steps minus one. Found and censored final updates remain included in the recorded episode lifecycle. The metadata records 49 found and 23 censored trajectories; these are collection descriptors, not an efficacy comparison.

The original parent ended with exit 1 after 131.561186667 seconds. It did not time out, reaped its child and recorded the process group absent. The worker reported no pending operations: 3 model loads, 74 resets, and 52,072 matched forward/step calls. The saved traceback identifies the fixed reservoir quota check during mixture preparation.

The original completed-collection auditor was not run. No replacement, resampling, seed change or retry was made. This readback opens only the plan, failure receipt, episode metadata and parent terminal. It does not decode checkpoint/NPZ arrays or replay trajectories. Raw numerical and detailed operation-journal correctness remains unaudited for this failed collection.

Input identities and all 72 per-cell episode memberships are recorded in summary.json. Original source and output files are unchanged.
