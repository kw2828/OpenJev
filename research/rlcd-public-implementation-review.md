# What the public Qwen RLCD implementation establishes

Source review, September 23, 2026. The supplied screenshot describes parallel
bounded decisions. The linked repository remains at revision `2af8684`.
This review inspected public source without downloading weights, running
inference, or reproducing the reported timings. It does not verify TypeSafe's
architecture or training method.

## Inference code, not a new trained checkpoint

The [file tree](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/tree/main)
contains application source, with no model weights or training pipeline.
The [engine router](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/engine.py)
selects MLX or PyTorch; its RLCD alias invokes parallel inference. The repository
name therefore does not establish a separately trained RLCD checkpoint.

The [MLX engine](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/engine_mlx.py)
loads Qwen2.5-1.5B-Instruct-4bit, prefills shared context, replicates KV state across
fields, batches field suffixes, and selects candidate entries from full-vocabulary
logits. Ordinary scoring applies restricted softmax with default temperature 1.
No fitted calibration procedure or calibration evaluation accompanies it.

For token collisions, the same engine greedily continues up to four tokens,
heuristically matches a candidate, and can fall back to the first choice. It
floors the selected confidence at 0.75 and spreads the remainder uniformly.
Those values are not demonstrated calibrated probabilities. Its telemetry still
reports one pass and zero generated tokens despite possible continuation calls.

The [PyTorch engine](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/engine_torch.py)
reads the collision flag but lacks that continuation branch. Candidates sharing
a compiled first-token ID receive identical scores. The backends consequently
do not implement equivalent collision semantics.

## Speed and decision-quality limits

The [schema compiler](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/schema.py)
uses independent field suffixes, without a dependency graph or conditioning on
other selected answers. Valid JSON does not establish joint consistency.
For fields exceeding 50 candidates, its baseline prompt lists only 20; the
parallel catalog supplies field descriptions without enumerating candidates.
The [baseline prompt](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/prompt_builder.py)
also requests indented multiline JSON. These different workloads do not isolate
cache reuse. The [benchmark runner](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/blob/main/core/benchmark.py)
performs one comparison per preset after warmup, rather than repeated latency
distributions or a labeled quality evaluation.

OpenJev already uses unique candidate labels and explicitly uncalibrated scores.
Its [shared-prefix study](shared-prefix-results.md) reports a 2.08x median paired
speedup for four-question synthetic requests, while single-question caching was
slower. All 1,296 records passed the fixed output checks, but exceptionally high
margins limit conclusions about difficult decisions. This is not a replication
of the upstream autoregressive-JSON comparison or evidence of better accuracy.

## A separate recurrent-memory hypothesis

A useful extension would update keyed memory only when an actual observation
arrives, correcting stale predictions before subsequent decisions. Compare it
with ordinary GRU, held-observation, additive-memory and memory-reset controls,
including total computation. The upstream code supports shared-context inference
as an engineering baseline; it provides no evidence that recurrent memory,
connectome wiring, world models or RL improve outcomes.
