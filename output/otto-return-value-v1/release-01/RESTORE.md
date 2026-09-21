# Restore the scalar-return study evidence

This archive preserves 50 original paths: all 44 worker payloads plus its
receipt, and the frozen plan, root execution witness, supervisor launch, log
and terminal. It includes every final checkpoint (nine), prediction archive
(nine), TRAIN/VALID dataset, public trajectory and operation journal.
No scientific success is implied by successful archive verification.

1. Download `openjev-otto-return-value-v1.tar.gz`, `manifest.json`,
   `SHA256SUMS.txt` and this `RESTORE.md` from the same release.
2. In that directory run `shasum -a 256 -c SHA256SUMS.txt`. Both the archive
   and its manifest must match; the sums also cover this document.
3. Use a fresh checkout, preserving any existing experiment directories:

```sh
git clone https://github.com/kw2828/OpenJev.git OpenJev-return-value
cd OpenJev-return-value
git checkout 8879b92ed9ce7586b4fae3bc8bf0eb07bb9a82dd
tar -xzf /absolute/path/to/openjev-otto-return-value-v1.tar.gz
```

The archive contains regular files under `output/otto-return-value-v1/` only;
extract at the repository root. Do not extract over an existing `run-01`.
The source freeze is `8879b92ed9ce7586b4fae3bc8bf0eb07bb9a82dd`. The plan SHA256 is `92df71d5e20ca48d8485bfa0a5f50bdf0e6e7a276c8e0c097bccd8e1c095aee2`
and binds 162 source files. Subsequent publication helpers are outside that
scientific source closure. Original absolute runtime/request paths remain
unchanged in receipts; moving the archive does not rewrite those witnesses.

`manifest.json` maps each original path to its exact SHA256 and byte count.
Verify all 50 restored files against that mapping before numerical reuse.
The publisher performs streaming per-member readback, gzip integrity reading
and checks that the originals did not change. No arrays or models are loaded.

Audit receipts and reports are published separately as small repository
evidence; their identities are bound in the archive manifest. This archive
is the complete new worker run, not a bundle of all historical inputs.
Re-running lineage authentication also requires the earlier study artifacts
named by the plan and the pinned runtime. Byte verification itself requires
neither those earlier datasets nor NumPy, Torch, a model or a simulator.
