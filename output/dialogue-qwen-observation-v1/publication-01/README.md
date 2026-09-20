# Frozen request packet

The 116,762,047-byte request packet exceeds GitHub's single-file limit. This gzip is a lossless delivery copy, verified by streaming decompression and SHA256. The original preparation manifest remains unchanged.

From the repository root, restore the exact file before reproducing scoring:

```sh
gzip -dc output/dialogue-qwen-observation-v1/publication-01/requests.jsonl.gz > output/dialogue-qwen-observation-v1/preparation-02/requests.jsonl
shasum -a 256 output/dialogue-qwen-observation-v1/preparation-02/requests.jsonl
```

Expected SHA256: `3c091e1e168a80f1f58dedbeb2e327fadce6142dd7a7c2d06aa0edca40466c2d`. Do not add this compressed copy inside the preparation directory, whose file membership is exact.
