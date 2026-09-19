# Replaying the dialogue memory factors

This diagnostic replays existing checkpoints from the completed
[candidate-memory study](dialogue-copy-reproduction.md). It does not fit a new
model. See the [fixed protocol](dialogue-evidence-qualification-protocol.md).

Use the original pinned runtime and source files, feature packet, lexical
cache, and all 15 completed fits. The qualifier authenticates the full old
study, then replays scalar and selective models for seeds 4101, 4102 and 4103.
The original individual data, predictions and weights are local artifacts;
the public aggregate report alone is insufficient to run this replay.

From the repository root, choose fresh output directories:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/qualify_dialogue_evidence.py freeze \
  --old-study runs/dialogue-copy-v1/study-01 \
  --packet runs/sgd-state-v1/features-02 \
  --lexical runs/dialogue-copy-v1/lexical-01 \
  --out output/dialogue-evidence-qualification-reproduction/protocol \
  --wall-cap-seconds 300
```

Use the emitted plan digest in the replay command:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/qualify_dialogue_evidence.py replay \
  --plan output/dialogue-evidence-qualification-reproduction/protocol/plan.json \
  --plan-sha256 REPLACE_WITH_EMITTED_DIGEST \
  --out runs/dialogue-evidence-qualification-reproduction/replay-01
```

Outputs are exclusive. A failed replay retains its failure and available
partial rows; do not replace it with an unreported retry. Every successful
fit must reproduce all original row identities and choices, with maximum
candidate-probability deviation no larger than `1e-6`. Bitwise agreement is
reported separately.

`operators.npz` retains the prior belief, writer distribution, departure mass,
actual result and scored-row metadata. Each fit's `diagnostic.json` reports
the prescribed partitions and one-step substitutions. Forced writing and
normalized-product substitutions do not feed back into subsequent states.
Their repair and harm counts therefore describe a diagnostic, not the
gameplay or accuracy of a newly trained model. Official test contents remain
outside this experiment.
