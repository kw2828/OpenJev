# Corrected memory replication

The [V2 protocol](dialogue-copy-v2-protocol.md) fits all 15 models anew with
normalized recurrent state. It does not resume the failed qualification or
repair old weights. The original feature packet and lexical cache are reused;
there is no new encoding or external model call.

This exact campaign requires the authenticated local feature and lexical
artifacts described in [the original reproduction guide](dialogue-copy-reproduction.md).
The public V1 execution metadata provides the original initialization hashes;
old trained weights are not used to initialize V2. This is an exact-artifact
replication command, not a tested clean-install or cross-platform recipe.
Freshly regenerated input receipts require their own declared experiment
identity rather than being relabeled as our recorded artifacts.

Run from the repository root with the existing Python runtime. Choose an
unused destination, freeze, and retain the returned plan digest:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/study_dialogue_copy_v2.py freeze \
  --packet runs/sgd-state-v1/features-02 \
  --lexical runs/dialogue-copy-v1/lexical-01 \
  --old-study output/dialogue-copy-v1/execution-01 \
  --out runs/dialogue-copy-v2/study-01
```

Publish and inspect the fixed plan before training. Use its actual digest:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
PYTHONPATH=src:scripts .venv/bin/python scripts/study_dialogue_copy_v2.py train \
  --packet runs/sgd-state-v1/features-02 \
  --lexical runs/dialogue-copy-v1/lexical-01 \
  --old-study output/dialogue-copy-v1/execution-01 \
  --out runs/dialogue-copy-v2/study-01 \
  --plan-sha256 REPLACE_WITH_FROZEN_PLAN_DIGEST
```

The 3,600-second cap covers the entire training/evaluation command. Failure
stops the attempt and preserves evidence. Do not reuse its directory, replace
a fit, or silently rerun it. Keep console logs outside the strict study folder.

Only after a successful completion, hash its `completed.json` and generate
the saved-output report in a fresh directory:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/report_dialogue_copy_v2.py \
  --run runs/dialogue-copy-v2/study-01 \
  --packet runs/sgd-state-v1/features-02 \
  --lexical runs/dialogue-copy-v1/lexical-01 \
  --plan-sha256 REPLACE_WITH_FROZEN_PLAN_DIGEST \
  --completed-sha256 REPLACE_WITH_COMPLETION_DIGEST \
  --historical-summary output/dialogue-copy-v1/report-01/summary.json \
  --historical-summary-sha256 4f895654bd5b046dee8a3c6e0a2ef7156b15a6299341450ebcae25398578b2f9 \
  --out output/dialogue-copy-v2/report-01
```

The report checks all 15 fits, internal-state coverage and the original 13
scientific criteria. It produces a comparison chart and retains every seed.
Technical validity and scientific effectiveness are separate outcomes. V1
deltas are descriptive and do not enter the continuation rule.
