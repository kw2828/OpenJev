# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub==1.31.0", "pyarrow==25.0.1"]
# ///
"""Download pinned public inputs without changing the project's frozen uv.lock."""
import argparse
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    boolq_rev = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
    for split in ("train", "validation"):
        file = hf_hub_download("google/boolq", f"data/{split}-00000-of-00001.parquet",
                               repo_type="dataset", revision=boolq_rev)
        rows = pq.read_table(file).to_pylist()
        (args.out / f"boolq-{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    clinc_rev = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
    for remote, local in (("data_full.json", "clinc.json"), ("domains.json", "domains.json")):
        url = f"https://raw.githubusercontent.com/clinc/oos-eval/{clinc_rev}/data/{remote}"
        with urllib.request.urlopen(url, timeout=60) as response:
            (args.out / local).write_bytes(response.read())
    snapshot_download("sentence-transformers/all-MiniLM-L6-v2",
                      revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                      "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"])
    print(f"Pinned inputs ready in {args.out.resolve()}")


if __name__ == "__main__":
    main()
