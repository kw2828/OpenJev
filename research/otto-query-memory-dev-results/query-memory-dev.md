# OpenJev query memory: DEV FAIL

Candidate: `trace_delta`. **6/13 conditions passed.**

Teacher-score gaps on fixed collector paths, with equal originating-case and fit-seed weighting. Lower is better. This does not establish autonomous performance, calibration, total-compute savings or architecture novelty.

All eight methods and all three fit seeds are shown. The candidate is fixed; an alternative method is not promoted after seeing results.

![Every method and fit seed on all DEV panels](query-memory-dev.png)

## lambda3 / P4 / later nonqueries

Supported originating cases: 3/3. Every collector path remains in the denominator.

| Method | Mean gap | 309000001 | 309000002 | 309000003 |
| --- | ---: | ---: | ---: | ---: |
| `pretrained` | 0.085460016918609341 | 0.10487228170241451 | 0.067744612022617778 | 0.083763157030795718 |
| `joint_aux` | 0.076030913090622848 | 0.055133218277488856 | 0.057818464290948557 | 0.11514105670343112 |
| `last_error` | 0.12267151705549061 | 0.15855995236638604 | 0.09793846013021669 | 0.11151613866986908 |
| `instant_delta` | 0.08022706526857766 | 0.098546354163535632 | 0.068465901680999267 | 0.073668939961198068 |
| `trace_delta` (candidate) | 0.11719881024617479 | 0.15506553648285068 | 0.092084355244981334 | 0.10444653901069235 |
| `trace_additive` | 0.23953555185328082 | 0.25078569053228894 | 0.26850653871269209 | 0.19931442631486149 |
| `trace_scrambled` | 0.089199764967471465 | 0.096540085982474197 | 0.071312721161162454 | 0.099746487758777758 |
| `trace_no_write` | 0.085460016918609341 | 0.10487228170241451 | 0.067744612022617778 | 0.083763157030795718 |

## lambda3 / P4 / full nonqueries

Supported originating cases: 3/3. Every collector path remains in the denominator.

| Method | Mean gap | 309000001 | 309000002 | 309000003 |
| --- | ---: | ---: | ---: | ---: |
| `pretrained` | 0.060249273320845904 | 0.064926758023916911 | 0.055981058276606051 | 0.059840003662014751 |
| `joint_aux` | 0.057812354602025974 | 0.042780370766844211 | 0.060287640371230994 | 0.07036905266800271 |
| `last_error` | 0.089354805500387491 | 0.10435484525070048 | 0.081990225419628401 | 0.081719345830833609 |
| `instant_delta` | 0.05566410272674216 | 0.059384493157173555 | 0.056602735270481395 | 0.051005079752571521 |
| `trace_delta` (candidate) | 0.082974548987693894 | 0.10127738313390396 | 0.073309299684884185 | 0.074336964144293555 |
| `trace_additive` | 0.2099330280762918 | 0.20355301777810877 | 0.25635294673389403 | 0.16989311971687263 |
| `trace_scrambled` | 0.062318836030186446 | 0.057643706667250737 | 0.059088960887858054 | 0.070223840535450541 |
| `trace_no_write` | 0.060249273320845904 | 0.064926758023916911 | 0.055981058276606051 | 0.059840003662014751 |

## lambda4 / P4 / later nonqueries

Supported originating cases: 3/3. Every collector path remains in the denominator.

| Method | Mean gap | 309000001 | 309000002 | 309000003 |
| --- | ---: | ---: | ---: | ---: |
| `pretrained` | 0.065956076495786969 | 0.034854648899990681 | 0.10166311894805564 | 0.061350461639314575 |
| `joint_aux` | 0.093039077669528347 | 0.060568980680890901 | 0.16638761229138374 | 0.052160640036310407 |
| `last_error` | 0.043059131697745084 | 0.04357422904822067 | 0.041214708517565209 | 0.044388457527449375 |
| `instant_delta` | 0.052480753987757943 | 0.033721948728707811 | 0.090779994038990341 | 0.032940319195575662 |
| `trace_delta` (candidate) | 0.036916700055385721 | 0.037951583248321416 | 0.034269049863143927 | 0.038529467054691811 |
| `trace_additive` | 0.17086089198065124 | 0.15120507163647454 | 0.19147738977736609 | 0.16990021452811313 |
| `trace_scrambled` | 0.065779541136934044 | 0.043008443792930257 | 0.097294999791220013 | 0.05703517982665187 |
| `trace_no_write` | 0.065956076495786969 | 0.034854648899990681 | 0.10166311894805564 | 0.061350461639314575 |

## lambda4 / P4 / full nonqueries

Supported originating cases: 3/3. Every collector path remains in the denominator.

| Method | Mean gap | 309000001 | 309000002 | 309000003 |
| --- | ---: | ---: | ---: | ---: |
| `pretrained` | 0.052201858072358026 | 0.034208216284937483 | 0.060947489739843812 | 0.061449868192292777 |
| `joint_aux` | 0.065753090312977006 | 0.061468634006651833 | 0.085434439058207814 | 0.050356197874071397 |
| `last_error` | 0.041983676787181325 | 0.042009659204270904 | 0.039584558313655503 | 0.044356812843617567 |
| `instant_delta` | 0.0388315898610943 | 0.032692202333032411 | 0.049998399401906084 | 0.033804167848344398 |
| `trace_delta` (candidate) | 0.036202748282075974 | 0.036315358993516643 | 0.033241673720782856 | 0.039051212131928408 |
| `trace_additive` | 0.16871058411551329 | 0.14805915871314038 | 0.18887081168136308 | 0.16920178195203639 |
| `trace_scrambled` | 0.051770657121672815 | 0.041631601490605089 | 0.056559030418036747 | 0.057121339456376617 |
| `trace_no_write` | 0.052201858072358026 | 0.034208216284937483 | 0.060947489739843812 | 0.061449868192292777 |

## Every continuation condition

| Condition | Result | Saved details |
| --- | --- | --- |
| `technical_completion` | PASS | `{}` |
| `lambda3:P4:supported_cases` | PASS | `{"actual": 3, "required": 2}` |
| `lambda3:P4:later_gap_10pct` | **FAIL** | `{"best_control": 0.07603091309062285, "candidate": 0.11719881024617479, "controls": {"instant_delta": 0.08022706526857766, "joint_aux": 0.07603091309062285, "last_error": 0.1226715170554906, "pretrained": 0.08546001691860934}}` |
| `lambda3:P4:full_gap_nonregression` | **FAIL** | `{"candidate": 0.0829745489876939, "controls": {"instant_delta": 0.05566410272674216, "joint_aux": 0.057812354602025974, "last_error": 0.08935480550038749, "pretrained": 0.060249273320845904}}` |
| `lambda3:P4:seed_309000001_nonregression` | **FAIL** | `{"candidate": 0.15506553648285068, "controls": {"instant_delta": 0.09854635416353563, "joint_aux": 0.055133218277488856, "last_error": 0.15855995236638604, "pretrained": 0.10487228170241451}}` |
| `lambda3:P4:seed_309000002_nonregression` | **FAIL** | `{"candidate": 0.09208435524498133, "controls": {"instant_delta": 0.06846590168099927, "joint_aux": 0.05781846429094856, "last_error": 0.09793846013021669, "pretrained": 0.06774461202261778}}` |
| `lambda3:P4:seed_309000003_nonregression` | **FAIL** | `{"candidate": 0.10444653901069235, "controls": {"instant_delta": 0.07366893996119807, "joint_aux": 0.11514105670343112, "last_error": 0.11151613866986908, "pretrained": 0.08376315703079572}}` |
| `lambda4:P4:supported_cases` | PASS | `{"actual": 3, "required": 2}` |
| `lambda4:P4:later_gap_10pct` | PASS | `{"best_control": 0.043059131697745084, "candidate": 0.03691670005538572, "controls": {"instant_delta": 0.05248075398775794, "joint_aux": 0.09303907766952835, "last_error": 0.043059131697745084, "pretrained": 0.06595607649578697}}` |
| `lambda4:P4:full_gap_nonregression` | PASS | `{"candidate": 0.036202748282075974, "controls": {"instant_delta": 0.0388315898610943, "joint_aux": 0.065753090312977, "last_error": 0.041983676787181325, "pretrained": 0.052201858072358026}}` |
| `lambda4:P4:seed_309000001_nonregression` | **FAIL** | `{"candidate": 0.037951583248321416, "controls": {"instant_delta": 0.03372194872870781, "joint_aux": 0.0605689806808909, "last_error": 0.04357422904822067, "pretrained": 0.03485464889999068}}` |
| `lambda4:P4:seed_309000002_nonregression` | PASS | `{"candidate": 0.03426904986314393, "controls": {"instant_delta": 0.09077999403899034, "joint_aux": 0.16638761229138374, "last_error": 0.04121470851756521, "pretrained": 0.10166311894805564}}` |
| `lambda4:P4:seed_309000003_nonregression` | **FAIL** | `{"candidate": 0.03852946705469181, "controls": {"instant_delta": 0.03294031919557566, "joint_aux": 0.05216064003631041, "last_error": 0.044388457527449375, "pretrained": 0.061350461639314575}}` |

Failed conditions: `lambda3:P4:later_gap_10pct`, `lambda3:P4:full_gap_nonregression`, `lambda3:P4:seed_309000001_nonregression`, `lambda3:P4:seed_309000002_nonregression`, `lambda3:P4:seed_309000003_nonregression`, `lambda4:P4:seed_309000001_nonregression`, `lambda4:P4:seed_309000003_nonregression`.

The caller authenticates the source audit and original process closures separately. This renderer makes no new admission decision.
