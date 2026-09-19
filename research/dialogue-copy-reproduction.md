# Reproducing the dialogue copy comparison

This runs the fixed [five-model, three-seed protocol](dialogue-copy-protocol.md). It reuses the prior supplied-schema SGD task and frozen encoder, then prepares public lexical observations. The development split was already inspected. A reproduced improvement is not untouched confirmation or an official full-DST score.

Start with the runtime, pinned train/dev download, data preparation, MiniLM cache, and **encode step only** in [dialogue-memory reproduction](dialogue-memory-reproduction.md). Skip that document's old freeze/train/report commands. Those instructions explain licenses, prior failures, and platform limits. The observed runtime is Python 3.12.13, PyTorch 2.14.0, NumPy 2.5.3, Transformers 5.17.0, and Matplotlib 3.11.2. It is not a complete transitive lockfile; clean package installation has not been tested here.

Continue in the same shell with `DIALOGUE_RUN`, `DIALOGUE_PY`, and `dialogue_sha` defined as in that guide. Use fresh directories. Never replace a failed fit or reuse our recorded plan hash on a different machine. Do not modify frozen source between preparation and reporting.

```sh
export PYTHONPATH="$PWD/src:$PWD/scripts"
DIALOGUE_PACKET_SHA=$(dialogue_sha "$DIALOGUE_RUN/features/completed.json")
DIALOGUE_DATA_SHA=$(dialogue_sha "$DIALOGUE_RUN/data/completed.json")

"$DIALOGUE_PY" scripts/prepare_dialogue_copy.py \
  --packet "$DIALOGUE_RUN/features" --data "$DIALOGUE_RUN/data" \
  --packet-sha256 "$DIALOGUE_PACKET_SHA" --data-sha256 "$DIALOGUE_DATA_SHA" \
  --out "$DIALOGUE_RUN/lexical" \
  > "$DIALOGUE_RUN/lexical.log" 2>&1

"$DIALOGUE_PY" scripts/study_dialogue_copy.py freeze \
  --packet "$DIALOGUE_RUN/features" --lexical "$DIALOGUE_RUN/lexical" \
  --out "$DIALOGUE_RUN/copy-study" \
  > "$DIALOGUE_RUN/copy-freeze.log" 2>&1

DIALOGUE_PLAN_SHA=$(dialogue_sha "$DIALOGUE_RUN/copy-study/plan.json")

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
"$DIALOGUE_PY" scripts/study_dialogue_copy.py train \
  --packet "$DIALOGUE_RUN/features" --lexical "$DIALOGUE_RUN/lexical" \
  --out "$DIALOGUE_RUN/copy-study" --plan-sha256 "$DIALOGUE_PLAN_SHA" \
  > "$DIALOGUE_RUN/copy-train.log" 2>&1

DIALOGUE_COMPLETED_SHA=$(dialogue_sha "$DIALOGUE_RUN/copy-study/completed.json")

"$DIALOGUE_PY" scripts/report_dialogue_copy.py \
  --run "$DIALOGUE_RUN/copy-study" --packet "$DIALOGUE_RUN/features" \
  --lexical "$DIALOGUE_RUN/lexical" --plan-sha256 "$DIALOGUE_PLAN_SHA" \
  --completed-sha256 "$DIALOGUE_COMPLETED_SHA" --out "$DIALOGUE_RUN/copy-report" \
  > "$DIALOGUE_RUN/copy-report.log" 2>&1
```

Run each step only after the preceding step succeeds. The shell's `set -euo pipefail` from the linked guide stops on failure. Logs belong outside the strict study directory. Training saves all 15 final models and evaluations, with no checkpoint selection. The reporter authenticates all files and recomputes metrics from saved predictions without loading models. It produces `summary.json`, `report.md`, `comparison.png`, and `receipt.json`; a completed report can still show a failed continuation rule.

Raw text, embeddings, checkpoints, lexical arrays, and individual predictions stay under ignored `runs/`. Public artifacts contain aggregate results, source, and receipts. Shared encoder preparation, lexical preparation, fit time, and evaluation time are different costs. Batch evaluation time is not live request latency. Reuse of frozen previous metric helpers is explicit; the reporting implementation is not an independent reproduction of every pipeline stage.
