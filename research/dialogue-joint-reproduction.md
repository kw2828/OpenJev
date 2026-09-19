# Candidate-conditioned observation control

**Recorded status: the capacity measurement completed, but admission failed.**
The [saved result](dialogue-joint-capacity-results.md) projected 1,254.48 seconds
against the fixed 720-second threshold. Only preparation and the 512-text
probe ran. Full encoding, study freezing, training and scientific reporting
below are conditional interface documentation; they were not executed and
must not proceed using the rejected receipt. Existing attempt directories
must not be overwritten or reused.

This is the exact-artifact path for the [fixed protocol](dialogue-joint-protocol.md),
using the existing local runtime and authenticated SGD artifacts. It is not a
tested clean-install recipe. Raw dialogue strings, embeddings, individual
predictions and weights stay local. A new preparation identity must not be
presented as the recorded experiment.

The three preparation phases have separate exclusive destinations. First
hash the protocol and freeze public text layout and sources, without encoding:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/prepare_dialogue_joint.py freeze \
  --packet runs/sgd-state-v1/features-02 \
  --lexical runs/dialogue-copy-v1/lexical-01 \
  --data runs/sgd-state-v1/data \
  --packet-sha256 e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308 \
  --lexical-sha256 2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54 \
  --data-sha256 677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12 \
  --protocol research/dialogue-joint-protocol.md \
  --protocol-sha256 REPLACE_WITH_PROTOCOL_DIGEST \
  --device mps --out runs/dialogue-joint-v1/preparation-01
```

Publish the preparation plan and source before the fixed 512-text capacity
probe. Use the actual `plan.json` digest:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/prepare_dialogue_joint.py capacity \
  --plan runs/dialogue-joint-v1/preparation-01/plan.json \
  --plan-sha256 REPLACE_WITH_PREPARATION_PLAN_DIGEST \
  --out runs/dialogue-joint-v1/capacity-01
```

Inspect its completion receipt. A successful capacity measurement can still
deny full encoding. Proceed only when its fixed cost rule permits it:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/prepare_dialogue_joint.py encode \
  --plan runs/dialogue-joint-v1/preparation-01/plan.json \
  --plan-sha256 REPLACE_WITH_PREPARATION_PLAN_DIGEST \
  --capacity runs/dialogue-joint-v1/capacity-01 \
  --capacity-sha256 REPLACE_WITH_CAPACITY_COMPLETION_DIGEST \
  --out runs/dialogue-joint-v1/features-01
```

Only a successfully completed full cache permits study freezing. The runner
uses the unchanged packet/lexical inputs plus that exact cache and V2 receipts:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/study_dialogue_joint.py freeze \
  --packet runs/sgd-state-v1/features-02 \
  --lexical runs/dialogue-copy-v1/lexical-01 \
  --joint runs/dialogue-joint-v1/features-01 \
  --joint-completed-sha256 REPLACE_WITH_CACHE_COMPLETION_DIGEST \
  --old-study output/dialogue-copy-v2/execution-01 \
  --out runs/dialogue-joint-v1/study-01
```

Publish the study plan before fitting. Invoke the same runner with `train`
instead of `freeze`, the same arguments, and
`--plan-sha256 REPLACE_WITH_STUDY_PLAN_DIGEST`. Use the same fixed thread
environment as V2: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4`.
Keep console logs outside all strict execution directories.

After successful completion, the saved-output reporter takes `--run`,
`--packet`, `--lexical`, `--joint`, `--plan-sha256`, `--completed-sha256` and
an exclusive `--out`. It must authenticate all 12 fits before reporting.
Scientific failure is a valid completed result; a stopped execution cannot be
reported as complete. Preserve every failed or rejected attempt and do not
reuse its directory or silently alter its budget, data or recipe.
