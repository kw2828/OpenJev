# Restore the completed symmetry-head experiment

The [GitHub release](https://github.com/kw2828/OpenJev/releases/tag/otto-symmetry-head-v1) contains the lossless raw archive, its manifest and checksums. The archive is **244,249,285 bytes**, with SHA-256:

```text
34c18bf08d50ed875a72168d498d5010f4947e1d0162370886b7a30248c01158
```

Its 47 members contain all 42 original run files, the frozen plan, the execution witness, and the supervisor launch, log and terminal records. All twelve initial/final checkpoints, complete training arrays, trajectories and work records are retained. The independent audit, figures and post hoc diagnostic are in Git alongside this file. Downloading this evidence does not require running a model or simulator.

Use a fresh checkout so extraction does not overwrite local work:

```sh
git clone --branch otto-symmetry-head-v1 https://github.com/kw2828/OpenJev.git OpenJev-symmetry
cd OpenJev-symmetry
mkdir downloaded-symmetry-release
gh release download otto-symmetry-head-v1 --repo kw2828/OpenJev \
  --dir downloaded-symmetry-release \
  --pattern openjev-otto-symmetry-head-v1.tar.gz \
  --pattern manifest.json --pattern SHA256SUMS.txt
cd downloaded-symmetry-release
shasum -a 256 -c SHA256SUMS.txt
cd ..
tar -xzf downloaded-symmetry-release/openjev-otto-symmetry-head-v1.tar.gz
```

On Linux, `sha256sum -c SHA256SUMS.txt` is an equivalent checksum command. Compare the downloaded manifest and checksum file with the versions in this checkout as well. The archive must match the explicit hash above before extraction.

Verify every restored member against the published manifest:

```python
from pathlib import Path
import hashlib
import json

root = Path.cwd()
manifest = json.loads(
    (root / "output/otto-symmetry-head-v1/release-01/manifest.json").read_text()
)
assert len(manifest["members"]) == 47
for name, expected in manifest["members"].items():
    path = root / name
    assert path.is_file() and not path.is_symlink(), name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert path.stat().st_size == expected["bytes"], name
    assert digest.hexdigest() == expected["sha256"], name
print("Verified all 47 original members")
```

The frozen study source was committed and pushed at `bfe701f29e2eb5feb41bf15885a776d098b21859` before execution. The release tag adds results and saved-data presentation tools. The [plan](../plan-01.json) lists 121 scientific source pins, four inherited evidence inputs and the exact runtime identity. The [prior boundary-control release](https://github.com/kw2828/OpenJev/releases/tag/otto-boundary-control-v1) retains its separate raw history.

The original independent auditor authenticates absolute launch paths, the original interpreter and historical supervision records. It is deliberately tied to that execution environment; restoring bytes on another machine does not make its original command portable. Do not rewrite historical receipts or claim a fresh audit from byte verification alone. New portable analysis should read the restored arrays with `allow_pickle=False`, preserve the published split and weighting definitions, and record its own runtime and outputs.

The completed comparison fails its continuation rule: **2/54 conditions pass**. Restoring or replaying these files does not change that result.
