# Published scored artifacts

The [completed study release](https://github.com/kw2828/OpenJev/releases/tag/research-reacher-objective-ablation-v1)
contains all three scored archive parts and three verification sidecars. The
study and its independent saved-output audit completed; the frozen continuation
criterion **failed, with 27/31 checks passing**. Publication does not change that
decision.

The archive contains 5,023 files: the complete current execution, saved-output
audit, frozen sources and closed process records. It is 2,735,579,868 bytes,
split into three ordered parts. The [release verification](release-verification.json)
matches every scored asset's live GitHub SHA-256 and size against local bytes.
All three public sidecars were also downloaded and hashed. The large remote
parts were not downloaded again; the local parts were streamed in order to
verify the assembled archive.

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| [part00](https://github.com/kw2828/OpenJev/releases/download/research-reacher-objective-ablation-v1/scored-execution-and-audit.tar.gz.part00) | 943,718,400 | `a4ca263513cf760e92d96abecb4af95afcf9d1a0a15eb6cf82d52258ddcf61ee` |
| [part01](https://github.com/kw2828/OpenJev/releases/download/research-reacher-objective-ablation-v1/scored-execution-and-audit.tar.gz.part01) | 943,718,400 | `4c5f8aa6ab939fd1681073a2f1b5b97ee3d828c1432a423d433c7fb1b01a420c` |
| [part02](https://github.com/kw2828/OpenJev/releases/download/research-reacher-objective-ablation-v1/scored-execution-and-audit.tar.gz.part02) | 848,143,068 | `b871b7f1ad2a90138e5e7ffbbed5cda3666254cad54f6425ec9d700732ce1835` |

The release also includes [scored-SHA256SUMS](https://github.com/kw2828/OpenJev/releases/download/research-reacher-objective-ablation-v1/scored-SHA256SUMS),
[scored-manifest.json](https://github.com/kw2828/OpenJev/releases/download/research-reacher-objective-ablation-v1/scored-manifest.json)
and [scored-package-receipt.json](https://github.com/kw2828/OpenJev/releases/download/research-reacher-objective-ablation-v1/scored-package-receipt.json).
The latter two are byte-identical to [manifest.json](manifest.json) and
[receipt.json](receipt.json) here. The manifest binds every uncompressed member;
the receipt binds the archive, ordered parts and packager.

## Download and reassemble

Use a new download directory. These commands check every downloaded part and
sidecar, then check the complete archive before extracting into a new directory.
Archive paths begin at the repository root, including `runs/`, `evidence/`,
`scripts/` and `src/`.

```bash
gh release download research-reacher-objective-ablation-v1 --repo kw2828/OpenJev \
  --dir openjev-reacher-objective-release --pattern 'scored-*'
cd openjev-reacher-objective-release
shasum -a 256 -c scored-SHA256SUMS
cat scored-execution-and-audit.tar.gz.part00 \
  scored-execution-and-audit.tar.gz.part01 \
  scored-execution-and-audit.tar.gz.part02 > scored-execution-and-audit.tar.gz
printf '%s  %s\n' \
  'be2dc8aa33dd6392b6544fa48d15cf6cb4bf3fc0ee275c86fb6c8e40458d2109' \
  'scored-execution-and-audit.tar.gz' | shasum -a 256 -c -
mkdir restored-reacher-objective-ablation-v1
tar -xzf scored-execution-and-audit.tar.gz -C restored-reacher-objective-ablation-v1
```

## Lineage and retained failures

This archive is not a standalone copy of the complete research history. Full
lineage authentication also requires historical artifacts at the paths bound
by the [frozen protocol](../protocol/plan.json), including the original
[world-model study](https://github.com/kw2828/OpenJev/releases/tag/research-reacher-world-model-v1)
and [adaptive-search study](https://github.com/kw2828/OpenJev/releases/tag/research-reacher-search-v1).
Follow their restoration instructions for a complete re-audit; keep their
original receipts and failed-attempt boundaries intact.

The same release retains the separate
[engineering archive](https://github.com/kw2828/OpenJev/releases/download/research-reacher-objective-ablation-v1/engineering-whole-tree-v1.tar.gz)
and [preflight receipt](https://github.com/kw2828/OpenJev/releases/download/research-reacher-objective-ablation-v1/preflight.json).
Those describe synthetic implementation checks, not scored efficacy.

The two Python files are unchanged provenance snapshots of commands originally
run from `output/reacher-objective-ablation-v1/`, not executable entry points
from this evidence folder. The first packaging check incorrectly excluded
nested fit-completion receipts and stopped before creating an archive. Its
failure is retained in [first-packaging-failure.json](first-packaging-failure.json).
The corrected packaging attempt authenticated and reopened
every member. Neither packaging attempt called a model or changed the failed
scientific continuation criterion.
