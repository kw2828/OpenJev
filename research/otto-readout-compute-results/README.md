# OTTO readout versus recurrent adaptation at calibrated cost

**DEV_FAIL. Efficacy: 1/6 cells. Actual cost comparability: 3/3 seeds.**

The 116-parameter residual head trained for 74 epochs (666 updates); full joint trained 6,112 parameters for 40 epochs (360 updates). Both start from the same three parents and 54 TRAIN paths. Epoch counts were fixed before fresh collection, without live time-based stopping or intermediate checkpoint selection.

All nine views use the same 36 fresh DEV paths (14,338 rows), six cases per setting and three collectors. These paths were held out from training. This remains fixed-path development evidence, not autonomous performance, independent confirmation, statistical equivalence or an architecture novelty claim.

![Every fit, score, measured cost ratio, and efficacy check](methods.png)

Dots show all three fit seeds and diamonds their equal means. They share evaluation paths; no confidence interval or independent-data claim is inferred from these three fits.

## All nine views

| Arm | Fit seed | lambda3 full | lambda3 later | lambda4 full | lambda4 later | View seconds |
| --- | --- | --- | --- | --- | --- | --- |
| pretrained | 309000001 | 0.0798382926 | 0.0913428781 | 0.0904198849 | 0.113973803 | 1.28882021 |
| pretrained | 309000002 | 0.0850218004 | 0.0950800996 | 0.0990128359 | 0.119042352 | 1.31427192 |
| pretrained | 309000003 | 0.092632142 | 0.103958783 | 0.106483161 | 0.130698902 | 1.24318708 |
| action_residual_only | 309000001 | 0.0754457507 | 0.093023653 | 0.0738535204 | 0.0910002579 | 1.41962629 |
| action_residual_only | 309000002 | 0.0939268719 | 0.10398258 | 0.0950616968 | 0.107156073 | 1.26818879 |
| action_residual_only | 309000003 | 0.0685513677 | 0.0754695084 | 0.100753136 | 0.119571799 | 1.292058 |
| full_joint | 309000001 | 0.104673406 | 0.104631234 | 0.114528844 | 0.141549415 | 1.28443283 |
| full_joint | 309000002 | 0.0968813451 | 0.0938126016 | 0.103331224 | 0.123883102 | 1.37190583 |
| full_joint | 309000003 | 0.0720600833 | 0.0824941296 | 0.0875546326 | 0.0990349934 | 1.24039921 |

Lower case-weighted raw gap is better. Full includes every nonquery action; later includes nonquery steps >=5. Complete support and supplemental metrics remain in the audited JSON.

## Every paired comparison

| Seed | Setting | Supported later cases | Parent later/full | Residual later/full | Cell |
| --- | --- | --- | --- | --- | --- |
| 309000001 | lambda3 | 6/6 | FAIL / FAIL | FAIL / FAIL | FAIL |
| 309000001 | lambda4 | 6/6 | FAIL / FAIL | FAIL / FAIL | FAIL |
| 309000002 | lambda3 | 6/6 | FAIL / FAIL | PASS / FAIL | FAIL |
| 309000002 | lambda4 | 6/6 | FAIL / FAIL | FAIL / FAIL | FAIL |
| 309000003 | lambda3 | 6/6 | PASS / PASS | FAIL / FAIL | FAIL |
| 309000003 | lambda4 | 6/6 | PASS / PASS | PASS / PASS | PASS |

Each cell requires a >=5% later-gap reduction with strict improvement and no full-gap regression against both controls. A zero comparator gap cannot pass strict improvement. Overall continuation also requires all three actual time ratios within [0.90, 1.10].

## All six fit times and three cost ratios

| Seed | Residual74 seconds | Full-joint40 seconds | Residual / joint | Comparable |
| --- | --- | --- | --- | --- |
| 309000001 | 142.1225 | 133.8471 | 1.06182727 | PASS |
| 309000002 | 142.208501 | 141.369192 | 1.005937 | PASS |
| 309000003 | 142.439939 | 133.582818 | 1.06630434 | PASS |

Saved worker wall times: fresh collection 607.683951 s; producer 850.111827 s; independent audit 4.13017054 s. Fit timing includes construction, training, validation and checkpoint serialization in its recorded interval. Shared original pretraining and collection are excluded from fit ratios. These are measurements of the qualified CPU implementations, not optimal implementations or inference-speed benchmarks.

## Evidence

Original closure: [closure-01.json](<../../output/otto-readout-compute-v1/closure-01.json>) (`8ba451fb8f8e46c60e08d453f2b7be5808b3010f6af65bf33d8563e2b30da518`).

- plan: [registration-01.json](<../../output/otto-readout-compute-v1/registration-01.json>) (`f7af526bee016ad2e9ea1511c2850be0bfb00dc4df0823f96d89c2f46964d609`)
- producer receipt: [receipt.json](<../../output/otto-readout-compute-v1/training-01/receipt.json>) (`74ed92f5fac573cf76f4f765b886cbf608c3f7168732cad55affbf2153908127`)
- producer terminal: [training-native-01.terminal.json](<../../output/otto-readout-compute-v1/training-native-01.terminal.json>) (`360d011c4fdb1ed7ca65ecd1f86e159f0c394347231437a10be25ad5b91c5262`)
- audit receipt: [receipt.json](<../../output/otto-readout-compute-v1/audit-01/receipt.json>) (`016a1f4f613bdc3db30aa73e60635483423dbcb83e608d60f01b4db007ac8f7e`)
- audit terminal: [audit-native-01.terminal.json](<../../output/otto-readout-compute-v1/audit-native-01.terminal.json>) (`ce1f7c3ed84c16175b5bb1f8f124fa7f8d0cb2b7bc8e3efc6e332a5c6ffa3c79`)
- audit: [audit.json in the complete evidence archive](https://github.com/kw2828/OpenJev/releases/tag/otto-readout-compute-v1) (`d6dcc70752063520003b70a7c4ea73189029809695109a6e65ac2fba9a85879e`)

Both original process closures were authenticated. Publication reads saved JSON and opaque payload hashes, with no array/checkpoint decoding, model calls, optimizer updates or teacher/simulator calls. Prior studies remain closed; TEST and confirmation are unadmitted.
