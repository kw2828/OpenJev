# Expected-count prefix learning: synthetic diagnostic

All nine fits use the same 352-parameter recurrent model and paired original parameters. Only the public-prefix pretraining method changes; every arm then receives the same 480-epoch joint-training recipe.

![Every fit seed, blind regret and full measured training cost](benchmark.png)

[Frozen protocol](../finite-expected-count-learning-protocol.md) · [Full saved report](summary.json) · [Original closure and source receipt](receipt.json). Original successful process closures and all source and payload hashes were checked before reading result metrics.

## All unchanged criteria

| Criterion | Joint only | Gradient prefix | EM prefix |
|---|---|---|---|
| SHORT_HORIZON_LEARNING | FAIL (23/24) | PASS (24/24) | FAIL (22/24) |
| BLIND_EXTRAPOLATION | FAIL (11/21) | FAIL (15/21) | FAIL (13/21) |
| OBSERVED_FILTERING_EXTRAPOLATION | PASS (8/8) | PASS (8/8) | PASS (8/8) |

Every required condition must pass for every seed. SHORT uses H1/H2 cost MSE and regret at most half the positive uniform reference and observed KL at most 0.1 nats. BLIND uses H4/H8 cost thresholds and H8 survival MAE at most 0.05. OBSERVED uses H4/H8 KL at most 0.1. Required endpoint support is 256 TRAIN and 64 DEV. Favorable means or prefix likelihood cannot rescue failures; the full condition dictionaries remain in the summary.

## Longer-horizon means

| Arm | H | Blind regret | Blind MSE | Observed MSE | Observed KL |
|---|---:|---:|---:|---:|---:|
| joint_only | 4 | 0.296962 | 0.0644697 | 0.0647487 | 0.0929391 |
| joint_only | 8 | 0.325563 | 0.0700058 | 0.0658503 | 0.0588559 |
| gradient_prefix | 4 | 0.162069 | 0.0440527 | 0.0447894 | 0.0522711 |
| gradient_prefix | 8 | 0.20958 | 0.0473962 | 0.0452472 | 0.0431115 |
| em_prefix | 4 | 0.223469 | 0.049945 | 0.0497954 | 0.065009 |
| em_prefix | 8 | 0.230587 | 0.0508487 | 0.0472847 | 0.0507614 |

Means of three separately fitted policies on common cases; not an ensemble or significance test.

## Every paired EM comparison

Signed differences are EM minus control for the same seed and cases. Negative favors EM for that metric. These are descriptive differences, with no significance or additional continuation criterion.

| Control | Seed | H | Regret difference | Blind MSE difference | Observed MSE difference | KL difference |
|---|---:|---:|---:|---:|---:|---:|
| joint_only | 429261001 | 4 | -0.25956 | -0.0528037 | -0.0536726 | -0.0732989 |
| joint_only | 429261001 | 8 | -0.339045 | -0.0624186 | -0.060377 | -0.041431 |
| joint_only | 429261002 | 4 | +0.055682 | +0.00362196 | +0.000846819 | -0.0019237 |
| joint_only | 429261002 | 8 | +0.0418292 | +0.000129575 | -0.00597863 | +0.000514836 |
| joint_only | 429261003 | 4 | -0.0166004 | +0.00560763 | +0.00796606 | -0.00856766 |
| joint_only | 429261003 | 8 | +0.0122879 | +0.00481772 | +0.0106589 | +0.0166326 |
| gradient_prefix | 429261001 | 4 | -0.242224 | -0.0623456 | -0.0620857 | -0.0643007 |
| gradient_prefix | 429261001 | 8 | -0.309329 | -0.0663457 | -0.0640158 | -0.0538552 |
| gradient_prefix | 429261002 | 4 | +0.335805 | +0.0689601 | +0.0659293 | +0.0914604 |
| gradient_prefix | 429261002 | 8 | +0.33468 | +0.0705415 | +0.0619611 | +0.0562956 |
| gradient_prefix | 429261003 | 4 | +0.0906212 | +0.0110625 | +0.0111747 | +0.0110538 |
| gradient_prefix | 429261003 | 8 | +0.0376685 | +0.00616195 | +0.00816714 | +0.0205095 |

## Complete training costs and deadline outcomes

| Arm | Seed | Accepted / attempted | Timed stage s | Charged overrun s | Outside-stage summary s | Full prefix wrapper s | Construction s | Joint fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| joint_only | 429261001 | 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.001166 | 0.001775 | 27.921020 | 27.923960 |
| joint_only | 429261002 | 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.001058 | 0.000251 | 29.469813 | 29.471122 |
| joint_only | 429261003 | 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.001523 | 0.000390 | 29.606425 | 29.608339 |
| gradient_prefix | 429261001 | 1121 / 1122 | 10.002927 | 0.002927 | 0.003540 | 10.007952 | 0.000201 | 27.695390 | 37.703544 |
| gradient_prefix | 429261002 | 1048 / 1049 | 10.005071 | 0.005071 | 0.003157 | 10.010359 | 0.000438 | 29.516199 | 39.526996 |
| gradient_prefix | 429261003 | 935 / 936 | 10.010499 | 0.010499 | 0.004945 | 10.018223 | 0.000326 | 29.862701 | 39.881250 |
| em_prefix | 429261001 | 21 / 22 | 10.418817 | 0.418817 | 0.010941 | 10.434666 | 0.000230 | 29.071457 | 39.506353 |
| em_prefix | 429261002 | 20 / 21 | 10.311434 | 0.311434 | 0.003596 | 10.319126 | 0.000270 | 27.782688 | 38.102084 |
| em_prefix | 429261003 | 19 / 20 | 10.452214 | 0.452214 | 0.004976 | 10.459083 | 0.000379 | 29.679716 | 40.139178 |

The two active prefix methods receive 10 seconds of update eligibility, not identical total wall time or FLOPs. A complete update must finish within the window. Late work is rolled back, retained in the trace and charged; at least one accepted update is required. Full prefix-wrapper time includes boundary snapshots and final diagnostics. The first three timing columns are nested within that wrapper and are not added again to total training time.

The eligibility clock is monotonic `perf_counter`; the outer phase deadline is suspend-inclusive. Machine contention can change update counts. EM uses expected-count MAP updates; gradient pretraining uses persistent Adam. Both optimize the same likelihood plus 0.001 log-prior, with no cost-head updates. Joint training starts from the retained prefix state, uses a fresh Adam optimizer, and does not add the pretraining prior.

| Arm | Seed | Joint updates | Gradient passes | Expected-count passes | Prefix event NLL (DEV) |
|---|---:|---:|---:|---:|---:|
| joint_only | 429261001 | 3840 | 0 | 0 | 0.848538 |
| joint_only | 429261002 | 3840 | 0 | 0 | 0.845846 |
| joint_only | 429261003 | 3840 | 0 | 0 | 0.849468 |
| gradient_prefix | 429261001 | 3840 | 1122 | 0 | 0.854369 |
| gradient_prefix | 429261002 | 3840 | 1049 | 0 | 0.81924 |
| gradient_prefix | 429261003 | 3840 | 936 | 0 | 0.852405 |
| em_prefix | 429261001 | 3840 | 0 | 44 | 0.819576 |
| em_prefix | 429261002 | 3840 | 0 | 42 | 0.846956 |
| em_prefix | 429261003 | 3840 | 0 | 40 | 0.851898 |

Complete attempted-update traces, accepted and restored hashes, probability checks, diagnostic passes, event exposures, copies, hashes and rollback work remain in `pretraining` in the summary. The 21 common-training/evaluation structural-work routes are also retained; they are not exhaustive FLOP measurements.

## Data and whole-phase costs

| Split | Attempted prefixes | Endpoint eligible | Found-terminated | Valid prefix events |
|---|---:|---:|---:|---:|
| train | 512 | 481 | 31 | 4513 |
| base | 128 | 121 | 7 | 1124 |

The authenticated engineering log reports 156 passing tests. All nine final joint checkpoints were saved before DEV generation. Prefix likelihood includes every attempted history and its first found event, without post-terminal padding or replacement cases.

| Original closed phase | Seconds including launch and cleanup |
|---|---:|
| qualify | 5.644 |
| fit | 323.829 |
| audit | 2.207 |
| Sum of successful phases | 331.680 |

Whole-phase durations already contain their nested operations. The producer includes generation, exact-control checks, all training, evaluation and writes. Separate generation and inference timings remain in the summary. These are aggregate workload timings, not single-decision latency.

## Interpretation limits

This tests a learning recipe on a fixed synthetic environment, not a new architecture. All learners receive public tokens only, but share a privileged world-aligned initial cost head and known uniform reset prior. The exact correctness control alone receives the true boundary posterior; the uniform reference knows dynamics while discarding history. Observed filtering consumes intervening labels and is distinct from blind extrapolation.

The audit reconstructs targets and metrics and decodes 18 prefix-boundary checkpoints for hash/head/retained-state checks. It does not replay optimizer updates or independently establish historical timings; those remain source-qualified attestations. Joint-final checkpoints remain opaque. No latent-state identification, calibration, scenario-shift robustness, robotics or native transfer, novelty, or ICLR-readiness claim follows from this diagnostic. Earlier failed studies remain closed.

## Appendix: every endpoint result

All 36 audited arm/seed/horizon cells are shown, without filtering or seed selection.

| Arm | Seed | H | Cases | Blind regret | Blind cost MSE | Blind survival MAE | Observed cost MSE | Observed survival MAE | Observed KL | Shuffled blind regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| joint_only | 429261001 | 1 | 121 | 0.229583599 | 0.0502284761 | 0.00263562836 | 0.0502284761 | 0.00263562836 | 0.0323601186 | 0.592163765 |
| joint_only | 429261001 | 2 | 121 | 0.216316769 | 0.0512164066 | 0.0037558374 | 0.0572435431 | 0.00243376144 | 0.0613134665 | 0.557307657 |
| joint_only | 429261001 | 4 | 121 | 0.264087983 | 0.0563220729 | 0.00472786195 | 0.0575675199 | 0.00180950382 | 0.0848758519 | 0.570111382 |
| joint_only | 429261001 | 8 | 121 | 0.343535192 | 0.0668701451 | 0.00652790507 | 0.0638109707 | 0.00211450113 | 0.050056909 | 0.554054693 |
| joint_only | 429261002 | 1 | 121 | 0.183990556 | 0.0546642584 | 0.0026092173 | 0.0546642584 | 0.0026092173 | 0.0442034942 | 0.591260312 |
| joint_only | 429261002 | 2 | 121 | 0.240322411 | 0.0629235631 | 0.00356206725 | 0.0663603808 | 0.00219329027 | 0.072593273 | 0.573834959 |
| joint_only | 429261002 | 4 | 121 | 0.281019052 | 0.0665816542 | 0.0043702454 | 0.0676845137 | 0.00186961766 | 0.0980774779 | 0.595395676 |
| joint_only | 429261002 | 8 | 121 | 0.29332669 | 0.0710486895 | 0.00610603806 | 0.0694315346 | 0.00188870984 | 0.0615212968 | 0.509253465 |
| joint_only | 429261003 | 1 | 121 | 0.271584573 | 0.0622351182 | 0.00252246948 | 0.0622351182 | 0.00252246948 | 0.0670888528 | 0.558940587 |
| joint_only | 429261003 | 2 | 121 | 0.292775181 | 0.0653172889 | 0.00340039307 | 0.062101848 | 0.00246734092 | 0.0631402269 | 0.553698794 |
| joint_only | 429261003 | 4 | 121 | 0.345779292 | 0.0705053387 | 0.00487950035 | 0.0689940555 | 0.00204532197 | 0.0958638625 | 0.612666373 |
| joint_only | 429261003 | 8 | 121 | 0.339826147 | 0.0720987139 | 0.00620753942 | 0.0643083386 | 0.00223865179 | 0.0649896265 | 0.48480692 |
| gradient_prefix | 429261001 | 1 | 121 | 0.242823397 | 0.0565566522 | 0.00241437709 | 0.0565566522 | 0.00241437709 | 0.0472277772 | 0.562029597 |
| gradient_prefix | 429261001 | 2 | 121 | 0.205790569 | 0.0604383681 | 0.00338165133 | 0.0597472797 | 0.00254855069 | 0.0514778162 | 0.489138662 |
| gradient_prefix | 429261001 | 4 | 121 | 0.246752115 | 0.0658639677 | 0.00454182831 | 0.0659806134 | 0.00227398738 | 0.0758776645 | 0.497912933 |
| gradient_prefix | 429261001 | 8 | 121 | 0.31381896 | 0.0707972196 | 0.00652676225 | 0.0674497287 | 0.00233508575 | 0.0624810948 | 0.463298485 |
| gradient_prefix | 429261002 | 1 | 121 | 0.000577676553 | 0.00135632359 | 0.00270389083 | 0.00135632359 | 0.00270389083 | 0.00467246512 | 0.61136689 |
| gradient_prefix | 429261002 | 2 | 121 | 0.00130957988 | 0.00133292006 | 0.00356425451 | 0.00181933123 | 0.00261235746 | 0.00609269671 | 0.573092411 |
| gradient_prefix | 429261002 | 4 | 121 | 0.000895820449 | 0.00124349389 | 0.00490495243 | 0.00260206854 | 0.00208761682 | 0.00469337364 | 0.57081818 |
| gradient_prefix | 429261002 | 8 | 121 | 0.000475849781 | 0.000636774105 | 0.00624315407 | 0.00149183358 | 0.0021173702 | 0.00574054123 | 0.491349462 |
| gradient_prefix | 429261003 | 1 | 121 | 0.242884383 | 0.0563070274 | 0.00241997423 | 0.0563070274 | 0.00241997423 | 0.0464476227 | 0.569991755 |
| gradient_prefix | 429261003 | 2 | 121 | 0.212293086 | 0.0601977554 | 0.00335871183 | 0.0594280341 | 0.0025784507 | 0.0510089889 | 0.48885836 |
| gradient_prefix | 429261003 | 4 | 121 | 0.238557631 | 0.0650504888 | 0.00446961518 | 0.0657854486 | 0.00227730051 | 0.0762423629 | 0.497912537 |
| gradient_prefix | 429261003 | 8 | 121 | 0.314445486 | 0.0707544847 | 0.00645979345 | 0.0668000573 | 0.00236218383 | 0.0611127664 | 0.450622123 |
| em_prefix | 429261001 | 1 | 121 | 0.00441912652 | 0.00264565133 | 0.00259157591 | 0.00264565133 | 0.00259157591 | 0.00958771103 | 0.59135507 |
| em_prefix | 429261001 | 2 | 121 | 0.00563437393 | 0.00293393082 | 0.00366592401 | 0.00357599616 | 0.00250002205 | 0.010605917 | 0.572773787 |
| em_prefix | 429261001 | 4 | 121 | 0.00452796311 | 0.00351836226 | 0.00488936889 | 0.00389487677 | 0.0019901319 | 0.0115769495 | 0.566325937 |
| em_prefix | 429261001 | 8 | 121 | 0.00448994107 | 0.00445155413 | 0.00634505677 | 0.00343392145 | 0.00207790061 | 0.00862591658 | 0.496115069 |
| em_prefix | 429261002 | 1 | 121 | 0.288998262 | 0.0616792669 | 0.00244225523 | 0.0616792669 | 0.00244225523 | 0.0649603728 | 0.582464164 |
| em_prefix | 429261002 | 2 | 121 | 0.31969563 | 0.0654013823 | 0.00336213641 | 0.0622879769 | 0.00247743466 | 0.0633434699 | 0.57624066 |
| em_prefix | 429261002 | 4 | 121 | 0.336701085 | 0.0702036131 | 0.00472457592 | 0.0685313331 | 0.00201259078 | 0.0961537808 | 0.594955372 |
| em_prefix | 429261002 | 8 | 121 | 0.335155864 | 0.0711782646 | 0.00569516111 | 0.0634529062 | 0.00216115493 | 0.062036133 | 0.497609004 |
| em_prefix | 429261003 | 1 | 121 | 0.306566302 | 0.0654997488 | 0.00268366474 | 0.0654997488 | 0.00268366474 | 0.063352707 | 0.565274989 |
| em_prefix | 429261003 | 2 | 121 | 0.265077328 | 0.0704977251 | 0.00377639757 | 0.0685976492 | 0.00270966846 | 0.0643748152 | 0.533599114 |
| em_prefix | 429261003 | 4 | 121 | 0.32917886 | 0.0761129656 | 0.00517891538 | 0.076960112 | 0.00249822755 | 0.0872962055 | 0.590369675 |
| em_prefix | 429261003 | 8 | 121 | 0.352114031 | 0.0769164299 | 0.00681864152 | 0.0749671964 | 0.00243789147 | 0.0816222747 | 0.464234393 |
