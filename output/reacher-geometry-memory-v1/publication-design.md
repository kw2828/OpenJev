# Geometry-memory publication packaging design

**Prospective packaging design only.** No current scientific execution, audit, summary or control payload was read for this note. No archive, upload or release was created. The completed-study packager must remain disabled until the parent supplies explicit completion authorization and external final audit/report receipt hashes. A failed scientific continuation gate is still publishable; an incomplete or unauthenticated audit is not a completed bundle.

Study: `reacher-geometry-memory-v1`. Plan SHA256: `23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee`.

## Reuse the byte-handling machinery, replace study assumptions

The prior `output/reacher-geometry-score-v1/package.py` provides suitable pure helpers for strict JSON, safe paths, exact member manifests, streaming deterministic tar/gzip, per-member decompression verification, byte splitting and ordered-part reassembly hashing. Its publisher supplies draft-first upload and exact remote size/digest verification. Copy the small necessary helpers into new, independently hash-bound publication scripts, or import only an explicitly pinned, import-safe helper module. Never alter the old scripts or import a runner/auditor to perform packaging.

Replace all old study pins, paths and coverage logic. The prior six-fit/two-score-mode/48-diagnostic-root schema is wrong for this study. Do not reuse its retrospective-launch assumptions or its hard-coded engineering attempt paths. Bind the actual new readiness, launch, launch-confirmation and eventual terminal receipts as they exist, with their original timing descriptions.

## Scientific archive: exact coverage

Use an independent filename-only enumerator equivalent to the frozen new auditor's `expected_members`, checked against both the execution completion manifest and audit receipt. Expected execution membership is **9,505 files plus `completed.json`**, before adding source, audit and reporting artifacts:

| Scope | Required contents |
|---|---|
| Inherited inputs | All 74 original members: training data, three paired initialization/order sets, and all five original fit files for each of 12 fits |
| Parent lineage | Eight copied cache/geometry plan, audit, completion and summary documents; authenticate both parent chains |
| Model identity | Before/after tensor snapshots for all 12 fits and full restoration/final receipts |
| Fresh decisions | All 50 saved innovation pairs and the complete random-stream manifest |
| Learned rows | All 36 rows: native episodes, timing, executed predictions, real/carried state buffers and work, all 50 CEM traces and scoring diagnostics |
| Physics rows | All nine known-state/particle/kinematic rows: native episodes, timing, root estimates/public inputs, all 50 CEM traces and full four-bank plus selected-advance nominal journals, lifetime work and applicable observer receipts |
| Floors | All six zero/uniform rows, including episodes, timing and planning placeholders |
| Terminal/phase records | Exact frozen execution membership, completion receipt, phase boundaries and complete costs |

Include exact **90 frozen source bytes**, the plan/readiness files, original launch/supervision metadata and terminal records, all independent audit files, and final authenticated renderer/report/replay files. Include `LICENSE`, dependency declarations/lockfile, publication helper sources and explicitly supplied final prose files as `path=SHA256` arguments. No broad repository staging or recursive inclusion of unrelated output trees.

The final reporter schema is a pending dependency. Bind its exact output member list and source hash after review, not a guessed copy of the old seven-file schema. Require all four families, all three fits, all three panels, all five references and every one of the 25 checks. Preserve the audit's distinctions among executed native transitions, candidate nominal replay, selected nominal advances and observer work. Packaging copies scientific pass/fail unchanged and performs no inference, native replay or gate redesign.

## Separate engineering archive and explicit ancestry

Archive every original file in both new engineering trees, including every retained failed or successful attempt:

- `output/reacher-geometry-memory-rehearsal-v1/`
- `output/reacher-geometry-memory-capacity-v1/`

Include exact historical source snapshots, all raw rehearsal checkpoints/51 rows/audit, all capacity probes/replay, the separately bound `tests/reacher_geometry_memory_fixture.py`, capacity helpers, and readiness inputs such as component validation, fixture-preflight QA, capacity verification/design, readiness review and freeze review. Enumerate and authenticate each attempt explicitly before archiving. Do not silently substitute current source for a historical snapshot. Report missing historical bytes as a blocker or an explicit dependency limit. A synthetic invalid-input test receipt is not evidence that transient test directories were retained.

Preserve older failed preparation ancestors through an **authenticated dependency manifest** for the completed geometry-score release, rather than duplicating its large archives by default. Once that publication has a successful final verification receipt, bind its repository/tag/commit, release ID, verification receipt SHA, package receipt SHA, manifest SHA, ordered preparation archive assets, sizes and hashes. Include those small original sidecars and retain member identities for its failed capacity attempt and failed rehearsal audit. The implemented helper requires that verified prior release and its external receipt hash; it has no automatic local-archive fallback. If the prior release is unavailable, stop and preserve the packaging failure. Any fallback to complete local ancestral archives requires a separately authorized and reviewed explicit contract change before another packaging attempt. A broken link or unverified tag is not an acceptable substitute.

This yields a complete current-study verification bundle with declared historical dependencies, not a claim that every ancestral experiment or external software distribution is embedded. Preserve existing licenses and dataset/simulator provenance; do not silently add external model weights or third-party caches.

## Streaming, storage and verification

The capacity projection is roughly 26 GB of raw scientific artifacts. Recheck free space against actual measured input bytes, both archives, duplicate split parts and a failure reserve. The earlier 100-GiB launch floor is useful context, not a guarantee of packaging space after execution. Compression ratio and final part count must come from actual bytes.

Create an exclusive `output/reacher-geometry-memory-v1/publication-v1` only after authorization. Write a start receipt before validation, recording requested external hashes. Use sorted repository-relative regular-file paths, reject symlinks/path traversal/duplicates, deterministic tar metadata and gzip timestamp zero, and bounded streaming buffers. Rehash every input immediately before archive insertion. Reopen each complete archive, stream every member, and verify its path, header, size and uncompressed SHA256 against the manifest. Rehash originals again after packaging.

Split any archive above **1,400,000,000 bytes** into consecutive ordered segments of at most that size. These are not individually extractable tar files. Hash each segment, then read all segments in manifest order and require concatenated byte count/hash to equal the already decompression-verified archive. Retain the complete archive locally; publish only bounded segments or the unsplit archive. Never hold a complete archive in memory.

Create `manifest.json`, `receipt.json`, `README.md` and `SHA256SUMS`, binding all member classifications, helper hashes, exact inputs, archive/part identities and unchanged scientific gate. On any exception, retain partial archives/parts and a stage-specific failure receipt; invalidate any premature success receipt. No overwrite, clobber or automatic retry.

## Later release action

A separately authorized publisher should require external package/audit/notes hashes and an exact already-pushed commit in `kw2828/OpenJev`. Create only a draft at `research-reacher-geometry-memory-v1`, upload each declared asset once, and publish only after exact remote membership, upload state, size, SHA256 and resolved tag identity match. Preserve failed drafts and receipts. After publication, verify public sidecar downloads; distinguish that from large-part verification via GitHub-provided digests. No release action is authorized by this note.

Before invoking either helper, test synthetic corruption/missing-member cases, source and reporting drift, failed-gate preservation, namespace/path rejection, exclusive failure receipts, deterministic archive reopening and splitting at/below/above a tiny test threshold. These are byte-handling tests, not new scientific runs. The current 90 scientific files remain unchanged.
