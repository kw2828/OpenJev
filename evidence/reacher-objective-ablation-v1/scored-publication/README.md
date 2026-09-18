# Scored artifact packaging

The verified archive contains 5,023 files: the complete current execution,
independent saved-output audit, frozen sources and closed process records.
Historical lineage artifacts remain separate dependencies. The local archive
is 2,735,579,868 bytes, split into three ordered parts for GitHub publication.
Upload verification will be recorded separately; the package receipt alone is
not evidence that remote downloads are ready.

`manifest.json` binds each uncompressed member. `receipt.json` binds the whole
archive, parts and packager. Reassemble the three parts in numeric order before
checking the whole-archive hash and opening the tar archive.

The two Python files are unchanged provenance snapshots of commands originally
run from `output/reacher-objective-ablation-v1/`, not executable entry points
from this evidence folder. The first packaging check incorrectly excluded
nested fit-completion receipts and stopped before creating an archive. Its
failure is retained. The corrected packaging attempt authenticated and reopened
every member. Neither packaging attempt called a model or changed the failed
scientific continuation criterion.
