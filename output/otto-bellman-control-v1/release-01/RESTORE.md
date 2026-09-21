# Restore Bellman-control raw evidence

This archive preserves 100 original paths and bytes: 91 worker payloads and
receipt, the plan, root execution witness, three supervisor files, and three
independent audit files. It includes all continuation and target checkpoints,
target arrays, public trajectories, fit records and operation journals.
Archival success is not scientific success and does not revise any failed gate.

1. Download `openjev-otto-bellman-control-v1.tar.gz`, `manifest.json`, `SHA256SUMS.txt`, and `RESTORE.md`
   from the same release. Run `shasum -a 256 -c SHA256SUMS.txt` there.
2. Use a fresh checkout; do not overwrite an existing experiment directory:

```sh
git clone https://github.com/kw2828/OpenJev.git OpenJev-bellman-control
cd OpenJev-bellman-control
git checkout 8d560b9
tar -xzf /absolute/path/to/openjev-otto-bellman-control-v1.tar.gz
```

The source freeze is `8d560b9`; plan SHA256 is `f8e62494bc67d7859824900f4599ea4c1f890d0ec13d2a43ca1ae679e41aac38`.
Members are regular files under `output/otto-bellman-control-v1/`. Extraction
is relative to the repository root. `manifest.json` lists each original path,
SHA256 and byte count; verify every restored file against that mapping.
Absolute runtime paths in historical receipts are preserved, not rewritten.

The publisher streams every archived member back, reads the gzip trailer,
then checks originals and pinned sources again. No arrays, models, training
or simulator are executed. These checks establish byte preservation only.

Prior scalar-study inputs are separately required for lineage authentication
and numerical replay. This archive does not duplicate the earlier TRAIN/VALID
caches, checkpoints, their upstream evidence, or the pinned runtime. Obtain
those earlier releases and source artifacts named by the frozen plan before
replaying the full audit. Byte verification needs only the archive and its
manifest, with no model frameworks or earlier datasets.
