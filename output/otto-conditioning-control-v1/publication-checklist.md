# Conditioning-control publication checklist

**Prepared while the original worker is running.** This is publication planning, not a result or permission to read partial outcomes. Only frozen source, plans, completed qualification metadata and previous release metadata were inspected. No current worker payloads, arrays, model calls or simulator calls were inspected or executed.

## Completion prerequisites

- Require the original worker receipt to be `completed`, with all 504 episodes, no pending operations, and its original parent terminal completed with return code zero and absent process group. Preserve any failure; do not package a partial run as completed evidence.
- Run the frozen full saved auditor once with external plan, worker and parent SHA256 pins. Require its completed receipt, agreement, exact 10 worker payloads and both audit payloads. Its numerical replay is real computation and must not be labeled zero readouts.
- Invoke the reviewed [reporter](../../scripts/report_otto_conditioning_control.py), SHA256 `a535b1e8cec66f7e8afa9302bb7fbaf14c077ab0215a5b79e3d4f0e542613140`, with external plan/worker/parent/audit pins. Inspect the PNG and all 21 table cells before publishing copies. It must retain all 30 conditions and 18 descriptive gain1 competence results.

The current full plan is [plan-01.json](plan-01.json), SHA256 `8a8aa3bdca98e5d7bc0f0319ea80dcc982f1f6d90fb7b07848f8f547a5d88f0a`, with 209 source pins. The source freeze containing that plan is commit `ff7fd838fab60c66b880792752f7ee56be78e60e`. Publication helpers are additional sources, not retroactive members of the scientific freeze.

## Exact current evidence inventory

Build an explicit path-to-SHA256/byte manifest after completion, using repository-relative original paths. Do not recursively scoop the study directory: that can capture an active log, a partial archive, or unrelated follow-up notes.

| Group | Required original files |
|---|---|
| Full worker, 11 files | `run-01/{started.json,runtime.json,inference-setup.json,native-setup.json,summary.json,work-contexts.jsonl,work.jsonl,eval-transitions.jsonl,eval-episodes.jsonl,evaluation.jsonl,receipt.json}` |
| Full audit, 3 files | `audit-01/{started.json,summary.json,receipt.json}` |
| Report, 5 files | `report-01/{report.md,plotted-values.json,otto-conditioning-control.png,otto-conditioning-control.svg,receipt.json}` |
| Qualification worker, 9 files | `qualification-01/{started.json,runtime.json,inference-setup.json,preparation.json,parity.jsonl,work-contexts.jsonl,work.jsonl,summary.json,receipt.json}` |
| Qualification audit, 3 files | `qualification-audit-01/{started.json,summary.json,receipt.json}` |
| Plans and parent records, 8 files | Both `plan-01.json` and `qualification-plan-01.json`; all three `.launch.json`, `.log`, `.terminal.json` files for each `run-process-01` and `qualification-process-01` |
| Execution provenance | Existing `qualification-execution-01.json`, `run-dispatch-01.json`, plan-verification records, and the future final execution witness with actual session/chunk/exit and external pins |
| Engineering and seed history | Files listed by the frozen plan's source map, including their receipts/logs; both seed-ledger review 01 and corrected 02, without relabeling the preserved first attempt |

The first six rows total **39 distinct files** before additional execution/engineering provenance and model dependencies. Derive and record the final deduplicated count from this explicit union; do not assume the old archive's count.

Copy these unchanged dependencies at their original paths, using the full plan's exact descriptors:

- Six `output/otto-conditioning-v1/run-01/final-{gain1,gain53}-{10101,10102,10103}.npz` files: **1,996,679 bytes** combined. These are existing fitted heads; the autonomous worker creates no new weights.
- Three `output/otto-return-value-v1/run-01/kernel-lambda{3,4,5}.npz` files: **278,199 bytes** combined.
- Preserve all 209 pinned source files, including ignored upstream source files under `tmp/otto-source-review-01/`; a Git tag alone does not guarantee those files exist. Preserve the upstream license separately and pin its bytes. Include the new reporter and any packaging script with their own hashes.
- Include original conditioning `summary.json` and `preparation.json` plus its completed receipt/plan/audit/parent records for fitting costs, c0 and checkpoint provenance. To claim its complete closed payload evidence, include every file in that original receipt, not selected records alone.

## Honest restoration scope

The current evidence, six heads and three kernels support this study's saved presentation and numerical replay inputs. They are **not by themselves a self-contained historical authentication chain**. The unchanged auditor calls the producer's read-only authentication, which requires complete earlier closures; qualification replay additionally requires the original TRAIN/VALID caches and row metadata.

Prefer a dependency manifest listing exact earlier release assets and their manifest hashes, alongside this new archive. The existing conditioning archive is 13,070,613 bytes, SHA256 `65e8d636900de1b873b247f7dffb00e85e315c0955ec3330acbd98e64a90b086`; its own manifest explicitly requires earlier scalar/capacity dependencies. Do not imply that downloading it alone resolves all history.

The qualification plan names TRAIN/VALID NPZ caches of 170,443,229 and 33,549,363 bytes, plus row metadata of 2,087,245 and 412,957 bytes. These sizes are plan metadata, not new array reads. If the publication claims fully offline historical authentication, include the complete transitive input/payload closure and a dependency-completeness check. Otherwise state precisely which previous releases must be restored. Preserve absolute historical paths inside receipts; document relocation separately rather than rewriting evidence.

Byte verification and saved-file inspection are portable. Existing strict numerical auditors additionally require the recorded absolute repository, input, historical worker-output and interpreter path layout, plus the pinned runtime. Cloning and extracting elsewhere does not satisfy these identity checks even with every dependency present. A different layout needs a separately reviewed relocation adapter; none is supplied here. Do not rewrite historical plans, receipts or other evidence to make a new location appear old.

## Packaging and byte verification

Use the streaming `pack`, `verify_archive` and `unchanged` pattern from [archive_otto_bellman_control.py](../../scripts/archive_otto_bellman_control.py). The simpler [conditioning packager](../../scripts/package_otto_conditioning.py) reads each whole file and recursively enumerates its directory; do not reuse that approach for a worker allowed up to 6 GiB output.

1. Authenticate completed receipts, source closure and parent joins first, then freeze the explicit inventory. Reject symlinks, traversal, duplicate names and nonregular members.
2. Create a new exclusive archive directory under a separately declared packaging clock/RSS/output budget. Stream original bytes into a lossless archive; never modify worker/audit/report directories.
3. Stream every archived member back and compare original path, size and SHA256 against the closed manifest. Read through the gzip trailer. Rehash originals and pinned sources afterward. Preserve partial output and failure receipts on any discrepancy.
4. Emit `manifest.json`, `RESTORE.md`, `SHA256SUMS.txt` and packaging receipt. Record exact archive/member sizes and hashes; distinguish byte-preservation success from scientific gate success.
5. Determine actual compressed size only after closure. GitHub release assets must each be under 2 GiB; if necessary use ordered parts below that limit, with per-part and concatenated-stream hashes and explicit restoration order. Do not infer size from the 6 GiB uncompressed cap. [GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
6. Upload only after visual/numerical delivery review. Verify remote asset names, byte sizes and SHA256 digests; download and stream-check when needed. Record the actual upload/verification outcome in a separate publication receipt. [Release-asset API](https://docs.github.com/en/rest/releases/assets).

## Repository delivery and claims

Track source/protocol, compact plans/receipts/audited summary, readable result, PNG/SVG, restoration/dependency manifests and review witnesses. Keep large public trajectories, work journals, cached arrays and archives in release assets, not ordinary Git. GitHub warns above 50 MiB and blocks ordinary files above 100 MiB. [Large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

The result must distinguish raw counts from hit-weighted success/moves/costs; retain all three seeds, analytic results, all 30 candidate decisions and all 18 gain1 descriptive checks. Preserve the scalar **FAIL, 3/6 passed**. Explain lambda5 as a supplied unseen sensing kernel on the same grid. This is an ordinary conditioning comparison, not an architecture, memory or connectome claim.

Publish original fitting, qualification, full worker, nested parent, saved audit, and publication costs as separate scopes. Controller cost includes load/72 and learned module setup/432 plus per-search work. H=1/100/10000 is an accounting scenario, not extra searches; qualification/pairing/diagnostics are not silently inserted into the fitted-cost formula. No new training or inference is needed to package or render the completed evidence.
