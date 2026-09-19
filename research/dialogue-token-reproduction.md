# Shared token evidence: execution guide

The [recorded preparation](dialogue-token-preparation-results.md) passed its
capacity rule and independent cache audit. The full cache completed in 17.88
seconds. These cost and representation checks do not establish task quality.

This follows the [prospective protocol](dialogue-token-protocol.md) using the
existing authenticated local inputs. It is not a tested clean-install recipe.
Raw dialogue strings, token vectors and individual predictions remain local;
published receipts identify them by hash. Each destination is exclusive.

Freeze public context layout and source identities before encoding:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/prepare_dialogue_tokens.py freeze \
  --packet runs/sgd-state-v1/features-02 \
  --lexical runs/dialogue-copy-v1/lexical-01 \
  --data runs/sgd-state-v1/data \
  --packet-sha256 e4503c3dd63d28b74e91877dfc9c69b81f4b880e4f07d240cf8c08331fa0b308 \
  --lexical-sha256 2e3e528ae05e55a32aa812d25a9dd609d7878e4daff77130994b0a63c32dad54 \
  --data-sha256 677d37ea581b3165b0adc96a0a0daab74271e0434df2ad23ace9568537e2cd12 \
  --protocol research/dialogue-token-protocol.md \
  --protocol-sha256 REPLACE_WITH_PROTOCOL_DIGEST \
  --device mps --out runs/dialogue-token-v1/preparation-01
```

Publish the frozen preparation plan, protocol and sources, then measure only
the fixed first 512 unique contexts:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/prepare_dialogue_tokens.py capacity \
  --plan runs/dialogue-token-v1/preparation-01/plan.json \
  --plan-sha256 REPLACE_WITH_PREPARATION_PLAN_DIGEST \
  --out runs/dialogue-token-v1/capacity-01
```

A successful measurement does not itself permit full encoding. Inspect its
recorded admission decision. Only an admitted receipt permits the next phase:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/prepare_dialogue_tokens.py encode \
  --plan runs/dialogue-token-v1/preparation-01/plan.json \
  --plan-sha256 REPLACE_WITH_PREPARATION_PLAN_DIGEST \
  --capacity runs/dialogue-token-v1/capacity-01 \
  --capacity-sha256 REPLACE_WITH_CAPACITY_COMPLETION_DIGEST \
  --out runs/dialogue-token-v1/features-01
```

Study freezing requires a completed admitted cache. The runner uses the
unchanged packet, lexical features and V2 receipt lineage:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/study_dialogue_tokens.py freeze \
  --packet runs/sgd-state-v1/features-02 \
  --lexical runs/dialogue-copy-v1/lexical-01 \
  --tokens runs/dialogue-token-v1/features-01 \
  --tokens-completed-sha256 REPLACE_WITH_CACHE_COMPLETION_DIGEST \
  --old-study output/dialogue-copy-v2/execution-01 \
  --out runs/dialogue-token-v1/study-01
```

Publish the study plan and every source in its manifest before fitting. Then
use the same arguments with `train` instead of `freeze`, adding
`--plan-sha256 REPLACE_WITH_STUDY_PLAN_DIGEST`. Preserve the fixed environment:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4`.
Keep console logs outside strict artifact directories. Report all twelve fits
only after successful completion and saved-output authentication.

Do not overwrite, resume or retry an existing attempt. A changed protocol or
input identity defines different work and cannot replace the recorded outcome.
The earlier candidate-concatenation attempt remains rejected, independently
of whether this different representation qualifies.
