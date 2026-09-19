# Reproducing the dialogue-memory development comparison

This runs the fixed [development protocol](dialogue-memory-protocol.md): seven methods, three seeds, 12 epochs, final-fit evaluation on official development data only. It is a supplied-service/slot categorical subtask, not full dialogue state tracking. These commands create a separate reproduction; they do not resume, overwrite or replace the original study.

Run from the OpenJev repository root, at the source revision you intend to reproduce. Keep all frozen source files unchanged from preparation through reporting. Raw text, embeddings, weights and individual predictions stay under ignored `runs/`; publish aggregates and provenance rather than those files.

## Runtime and dependencies

The inspected research environment was `.venv`, with Python **3.12.13**, macOS **26.6.1 arm64**, PyTorch **2.14.0**, NumPy **2.5.3**, Transformers **5.17.0**, Hugging Face Hub **1.31.0**, Tokenizers **0.23.2**, Safetensors **0.8.0**, Matplotlib **3.11.2**, pytest **9.1.1** and Ruff **0.16.8**. `.venv-robotics` is not a substitute: Transformers was absent there. The repository's basic installation does not supply every dependency below.

For an independent environment, install Python 3.12 and `curl` first, then:

```sh
python3.12 -m venv .venv-dialogue-repro
.venv-dialogue-repro/bin/python -m pip install \
  'torch==2.14.0' 'numpy==2.5.3' 'transformers==5.17.0' \
  'huggingface-hub==1.31.0' 'tokenizers==0.23.2' 'safetensors==0.8.0' \
  'matplotlib==3.11.2' 'pytest==9.1.1' 'ruff==0.16.8'
```

These are observed installed versions, not a complete transitive lockfile. A clean installation and platform-specific wheel availability have not been tested by this documentation task. Setup needs package-index, GitHub and Hugging Face access. Use `PYTHONPATH=src`; an editable installation of OpenJev and its unrelated simulator dependencies is unnecessary for these scripts.

The encoder defaults to MPS and uses float32. Choose `DIALOGUE_DEVICE=cpu` below on a machine without supported Apple MPS. CPU encoding is accepted by the same code but may differ numerically or in time; it receives its own feature hashes. Training stays on CPU with the protocol's four threads regardless of the encoder device. Freeze and train must use the same Python, Torch, NumPy and platform identity; a different machine needs its own plan, not our recorded hashes.

## Fresh directories and pinned downloads

Use one shell session. The root directory must not exist. If any stage fails, retain it and its failure receipt; diagnose before deliberately starting a separately named attempt. There is no automatic retry or partial-fit resume.

```sh
set -euo pipefail
export PYTHONPATH="$PWD/src"
export DIALOGUE_PY="$PWD/.venv-dialogue-repro/bin/python"
export DIALOGUE_RUN="$PWD/runs/dialogue-memory-reproduction-01"
export DIALOGUE_DEVICE=mps  # Set cpu here if MPS is unavailable.

"$DIALOGUE_PY" - <<'PY'
import os
from pathlib import Path
Path(os.environ['DIALOGUE_RUN']).mkdir(parents=True, exist_ok=False)
PY

dialogue_sha() {
  "$DIALOGUE_PY" -c 'import hashlib, sys; f=open(sys.argv[1], "rb"); print(hashlib.file_digest(f, "sha256").hexdigest())' "$1"
}
```

Fetch the official SGD train/dev sources at `e852981ae34990f4358979625854259302feaa78`: **149 files**, comprising 127 training shards, 20 development shards and two schemas, approximately **504 MB** of JSON. No test dialogue contents are fetched. The fetcher verifies upstream Git blob identities and records SHA-256 hashes. It can reuse verified files already present at its legacy cache path `runs/sgd-state-v1/source/`; the new destination remains exclusive. Dataset license: [CC BY-SA 4.0](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/e852981ae34990f4358979625854259302feaa78/LICENSE.txt).

```sh
"$DIALOGUE_PY" scripts/prepare_sgd_state.py fetch \
  --source "$DIALOGUE_RUN/source" --workers 8 \
  > "$DIALOGUE_RUN/fetch.log" 2>&1

"$DIALOGUE_PY" scripts/prepare_sgd_state.py prepare \
  --source "$DIALOGUE_RUN/source" --out "$DIALOGUE_RUN/data" \
  > "$DIALOGUE_RUN/prepare.log" 2>&1

DIALOGUE_DATA_SHA=$(dialogue_sha "$DIALOGUE_RUN/data/completed.json")
```

Preparation excludes development transcripts duplicated in training and retains the exclusion record. It separates public text from labels and preserves reserved NOT_MENTIONED/DONTCARE identities. The study later selects the protocol's deterministic 2,048 training dialogue IDs before dropping conversations without categorical questions.

Cache exactly the six files needed by the frozen MiniLM encoder. Encoding itself uses `local_files_only=True`; it does not fill a missing cache silently. This is a model-weight download, approximately 91 MB for the weights plus tokenizer/configuration files, not an inference call:

```sh
"$DIALOGUE_PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = 'sentence-transformers/all-MiniLM-L6-v2'
revision = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
names = ['config.json', 'model.safetensors', 'tokenizer_config.json',
         'special_tokens_map.json', 'tokenizer.json', 'vocab.txt']
receipt = {'repository': repo, 'revision': revision, 'files': {}}
for name in names:
    path = Path(hf_hub_download(repo_id=repo, filename=name, revision=revision))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt['files'][name] = {'sha256': digest, 'bytes': path.stat().st_size}
with (Path(os.environ['DIALOGUE_RUN']) / 'cache-files.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
PY
```

The encoder rehashes these files before and after use. Its implementation uses frozen transformer states, masked mean pooling, nonoverlapping 254-content-token chunks for longer inputs, token-count-weighted chunk aggregation and final L2 normalization. No source text is silently truncated. This is the protocol's encoding recipe, not a call to the SentenceTransformer convenience API.

## Encode, freeze, train and report

Each command must finish successfully before the next. `features`, `study` and `report` must be fresh; `freeze` creates the study directory, and `train` then uses that exact frozen plan. The training command also evaluates each final model on development queries and writes the two deterministic references. It does not select checkpoints using development scores.

```sh
"$DIALOGUE_PY" scripts/study_dialogue_memory.py encode \
  --data "$DIALOGUE_RUN/data" --data-sha256 "$DIALOGUE_DATA_SHA" \
  --out "$DIALOGUE_RUN/features" --device "$DIALOGUE_DEVICE" \
  > "$DIALOGUE_RUN/encode.log" 2>&1

"$DIALOGUE_PY" scripts/study_dialogue_memory.py freeze \
  --packet "$DIALOGUE_RUN/features" --out "$DIALOGUE_RUN/study" \
  > "$DIALOGUE_RUN/freeze.log" 2>&1

DIALOGUE_PLAN_SHA=$(dialogue_sha "$DIALOGUE_RUN/study/plan.json")

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 \
"$DIALOGUE_PY" scripts/study_dialogue_memory.py train \
  --packet "$DIALOGUE_RUN/features" --out "$DIALOGUE_RUN/study" \
  --plan-sha256 "$DIALOGUE_PLAN_SHA" \
  > "$DIALOGUE_RUN/train.log" 2>&1

DIALOGUE_COMPLETED_SHA=$(dialogue_sha "$DIALOGUE_RUN/study/completed.json")

"$DIALOGUE_PY" scripts/report_dialogue_memory.py \
  --run "$DIALOGUE_RUN/study" --packet "$DIALOGUE_RUN/features" \
  --plan-sha256 "$DIALOGUE_PLAN_SHA" --completed-sha256 "$DIALOGUE_COMPLETED_SHA" \
  --out "$DIALOGUE_RUN/report" \
  > "$DIALOGUE_RUN/report.log" 2>&1

dialogue_sha "$DIALOGUE_RUN/report/receipt.json"
```

The report authenticates all 21 fit receipts, weights and predictions, both references and packet labels. It writes `summary.json`, `report.md`, `accuracy.png` and a hash-bound receipt. Reporting makes no neural calls and does not independently replay the encoder or model forecasts. A completed report may contain a failed development continuation screen; completion is not an efficacy claim.

Encoder preprocessing is a separate cost. Evaluation wall time includes batch construction and prediction storage, and the learned carry baseline replays prefixes separately per question. These timings are not single-request latency or evidence that this implementation beats an optimized incremental dictionary. Keep all fits, failed attempts and receipts; do not modify thresholds or overwrite outputs to turn a failure into a pass.
