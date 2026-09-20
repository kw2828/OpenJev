# Portable token-alignment metric recheck

The [portable checker](../scripts/recheck_dialogue_alignment.py) recomputes saved primary metrics without training, inference, encoders or the original cache locations. It reuses the [independent audit arithmetic](../output/dialogue-token-alignment-scientific-v1/audit-01/audit.py), authenticated against SHA256 `fc0f252a4ce12a092993bc9033dd5bb11250c6ed0522ec49d5e097eac62080b1`. Keep these two files at their repository-relative locations; the checkout itself can move.

This is a metric-only check. It covers all nine fits on held-out-service all, changed and retained rows: accuracy, raw-log NLL, Brier score, selected-candidate branch errors, the 22 behavioral checks and paired repairs/new errors. Counts and gate comparisons must agree exactly; floating metrics use the inherited numerical tolerances. It does not repeat corpus, training, checkpoint, cache, source-closure or runtime verification, assess calibration, or reproduce other descriptive subgroups. The main report's full technical admission is authenticated as a recorded claim. The experiment remains exposed TRAIN development with privileged previous state.

## Required files

Obtain the exact saved subset separately if it is not included with the release. Aggregate reports alone do not supply prediction arrays. The original experiment directory is `output/dialogue-token-alignment-scientific-v1/training-01`, not a directory under `runs/`.

In a directory chosen as `--run`, preserve these relative paths and original bytes:

- `completed.json`, `plan.json`, `evaluation-rows.jsonl`;
- `fits/{method}-{seed}/predictions.npz` for every method `flat_stratum`, `token_mean`, `token_aligned` and seed `6201`, `6202`, `6203`.

In a directory chosen as `--report`, supply the original `receipt.json` and `summary.json`. No checkpoint, update journal, reference predictions, raw dialogue, MiniLM weights, token cache or pooled/lexical feature file is needed. The external completion digest authenticates the original 44-file manifest, but this checker hashes only the 11 selected manifest payloads. It does not claim that omitted members were reverified.

The original completion SHA256 is `e7222147935b1f1604d31da83c940ed428ae3745dfddec374cfa64f7ddd1ce7c`. The completed main-report receipt SHA256 is `7591182a5c07c9cd4ee4035128b6c0900d7d9177ba89efccfe92bfd528c8a6c5`; it binds summary SHA256 `b4b8c1d3f6e9d1e98096c2684a5bca84b011496f89044e3f3eb31260f2fbbdc3`. Use these published pins rather than a bundle's self-declared hash. Do not rewrite saved absolute paths: this checker never follows them, and rewriting would break the original identities.

## Run

The synthetic portability tests used Python 3.12.13 and NumPy 2.5.3 on macOS. A POSIX Python environment is needed by the pinned helper's standard-library imports; no Torch, MPS or encoder installation is required. For a separate environment, from the repository root:

```sh
python3.12 -m venv .venv-metric-recheck
.venv-metric-recheck/bin/python -m pip install numpy==2.5.3
```

From the repository root, the command below uses the original relative directory layout. Substitute other input directories if the saved subset was relocated. Choose an output directory that does not exist and is outside the input directories.

```sh
.venv-metric-recheck/bin/python scripts/recheck_dialogue_alignment.py \
  --run output/dialogue-token-alignment-scientific-v1/training-01 \
  --completed-sha256 e7222147935b1f1604d31da83c940ed428ae3745dfddec374cfa64f7ddd1ce7c \
  --report output/dialogue-token-alignment-scientific-v1/report-01 \
  --report-receipt-sha256 7591182a5c07c9cd4ee4035128b6c0900d7d9177ba89efccfe92bfd528c8a6c5 \
  --out output/dialogue-token-alignment-scientific-v1/portable-review-01
```

The checker authenticates all selected files before decoding predictions, then writes `summary.json` and a source-bound `receipt.json`. `metric_agreement: true` means the selected published metrics and checks were reproduced; `continuation_passed` is the separate scientific outcome and can be false. Failures leave `failed.json` and any earlier evidence. There is a 60-second checked wall limit, a 32 MiB output limit and no automatic retry or overwrite.

The [synthetic preflight receipt](../output/dialogue-token-alignment-scientific-v1/portable-preflight-01/receipt.json) records 21 passing tests, clean lint, exact commands and unchanged source hashes. Tests include physically relocated files, deliberately missing original caches/checkpoints, hash corruption, reordering, incomplete membership and metric mismatches. No actual study predictions were opened for that preflight.

## Recorded verification

The first invocation used the incorrect `runs/dialogue-token-alignment-scientific-v1/training-01` path and failed before opening predictions. Its [failure receipt](../output/dialogue-token-alignment-scientific-v1/portable-01/failed.json) is preserved. A separately recorded invocation corrected only the input path, with unchanged checker, external pins and rule. The [portable-02 receipt](../output/dialogue-token-alignment-scientific-v1/portable-02/receipt.json) and [summary](../output/dialogue-token-alignment-scientific-v1/portable-02/summary.json) confirm metric agreement for all nine fits and the unchanged scientific FAIL, with 18 of 22 checks passing. Neither invocation reran training, evaluation or an encoder.
