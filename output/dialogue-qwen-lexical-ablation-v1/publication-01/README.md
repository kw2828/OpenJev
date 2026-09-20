# Exact no-lexical actor requests

`requests.jsonl.gz` expands byte-for-byte to the request payload described by
[the preparation plan](../preparation-01/plan.json). Decompress it as
`../preparation-01/requests.jsonl` before reproducing the saved-only checks.
The original file is kept outside Git to avoid a duplicate 89 MB text artifact.

These prompts derive from the Schema-Guided Dialogue dataset (SGD) and retain
its CC BY-SA 4.0 license. See [source and attribution](../../../research/dialogue-memory-reproduction.md)
and [the original request publication](../../dialogue-qwen-observation-v1/publication-01/README.md).
The source dialogues, candidate catalog and previous values are unchanged;
this experiment removes lexical feature lines and their explanatory text.
This publication does not relicense SGD or Qwen weights.

No current targets, predictions, or reviewer answers were added to actor prompts.
This is a prepared experiment artifact, not an accuracy result.
