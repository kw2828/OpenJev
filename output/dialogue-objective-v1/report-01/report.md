# Fresh objective comparison

Six fresh fits on historically exposed official TRAIN, with privileged gold previous values. Four fixed readouts per seed; no selection or new architecture claim. Earlier 18/22 FAIL is unchanged. Main authentication hashes all execution payloads including opaque weights, and decodes prepared/split metadata plus authenticated schema index/integer offsets for work reconstruction. No float cache, encoder, producer, model or checkpoint deserializer is loaded. Execution/initializer/normalization receipts are source-bound witnesses, not training replay. Historical metrics are pinned references, not contemporaneous equal-capacity fits; historical row-level paired repairs are unavailable.

Continuation **FAIL**, 6/13 checks. Objective evidence: False; practical evidence: False.

| Readout | Seed | Changed accuracy | Retained error | Overall NLL | Equal-service NLL |
|---|---:|---:|---:|---:|---:|
| stratum-original | 6201 | 0.795847751 | 0.076232565 | 0.412225928 | 0.405749940 |
| stratum-original | 6202 | 0.856401384 | 0.081204254 | 0.412880339 | 0.329798238 |
| stratum-original | 6203 | 0.851211073 | 0.068084519 | 0.356761959 | 0.298404583 |
| stratum-corrected | 6201 | 0.695501730 | 0.050131197 | 0.319145730 | 0.341814541 |
| stratum-corrected | 6202 | 0.795847751 | 0.053583759 | 0.292738538 | 0.261905690 |
| stratum-corrected | 6203 | 0.785467128 | 0.040878332 | 0.267081860 | 0.243651745 |
| uniform-original | 6201 | 0.775086505 | 0.038530590 | 0.226666602 | 0.231916357 |
| uniform-original | 6202 | 0.697231834 | 0.030796851 | 0.218793300 | 0.197886486 |
| uniform-original | 6203 | 0.801038062 | 0.049026378 | 0.273531399 | 0.249282673 |
| uniform-reweighted | 6201 | 0.830449827 | 0.055517194 | 0.299192943 | 0.281437385 |
| uniform-reweighted | 6202 | 0.757785467 | 0.046678636 | 0.264466790 | 0.219018942 |
| uniform-reweighted | 6203 | 0.842560554 | 0.073194310 | 0.380082064 | 0.312050161 |

All readouts, seed means, historical families, service effects and paired repair/harm counts are in summary.json.
No confidence interval, significance or architecture novelty claim. Tiny rare-category supports remain descriptive.

| Fit | Training seconds | Evaluation seconds | Fit seconds |
|---|---:|---:|---:|
| stratum-6201 | 736.991135 | 7.583903 | 744.579058 |
| uniform-6201 | 737.454960 | 7.699185 | 745.158108 |
| stratum-6202 | 738.580075 | 7.751346 | 746.335836 |
| uniform-6202 | 738.093978 | 7.828640 | 745.926707 |
| stratum-6203 | 738.867323 | 7.807089 | 746.678133 |
| uniform-6203 | 735.973215 | 7.582027 | 743.561192 |

Whole campaign: 4480.106301 seconds.
Whole includes authentication, initialization, training, evaluation and IO. Training/evaluation nest inside each fit and are not added again. Schema preparation is separate historical cost. RSS is process-lifetime high-water.
