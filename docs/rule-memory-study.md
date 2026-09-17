# English rule memory: frozen development experiment

**Status: protocol frozen before fitting. Results pending.** This tests whether repeated reads help on [RuleTaker](https://github.com/allenai/ruletaker), following the unsuccessful routing continuation tests. The [data audit](ruletaker-development.md) describes the exact English-only inputs and disjoint worlds.

## Comparison

| Head | Context access | Reads | Role |
| --- | --- | ---: | --- |
| Question only | None | 0 | Shortcut control |
| Mean context | Average sentence values | 1 | Non-attention control |
| Single read | Question-conditioned attention | 1 | Matched attention control |
| Recurrent 3 | Tied attention/update | 3 | Multi-step candidate |
| Recurrent 6 | Tied attention/update | 6 | Additional-compute candidate |
| Coverage 3 | Tied attention, accumulated-attention penalty | 3 | Corrective candidate |

Every English sentence and assertion receives a frozen, normalized MiniLM embedding. No sentence is truncated or retrieved away. Facts and rules remain separate memory slots, including duplicate sentences. Gold answers, source IDs, proof annotations and depth never enter the head. Outputs are uncalibrated probabilities over fixed `false` and `true` IDs. This experiment does not establish arbitrary-candidate transfer.

The query, key and value projections map 384 to 128 dimensions. Each attention step reads all unmasked sentences and applies a shared residual update and layer normalization. A shared classifier sees the original query, final state and their elementwise product. Coverage subtracts cumulative previous attention weights from the next attention logits, with coefficient one. It adds no parameters. Single-read and recurrent heads have identical initial tensors and parameter counts. The question-only and mean controls use fewer active parameters; the report counts that explicitly.

These are established building blocks: [end-to-end memory networks](https://arxiv.org/abs/1503.08895) already use repeated memory reads, and [coverage attention](https://arxiv.org/abs/1704.04368) already discourages repetition. This combination is a diagnostic candidate, not a claimed new architecture. The sentence encoder is a pretrained transformer.

## Frozen procedure

- Train on 19,809 questions in 2,000 worlds, annotated depth 0-2.
- Evaluate on 2,947 development questions at depth 0-2 and 1,941 at depth 3-5. The latter changes both depth and context distribution.
- Six heads, three seeds (17, 29, 43), 20 epochs each, batch size 256. AdamW learning rate 0.001, weight decay 0.0001, gradient norm cap 1. Same shuffled question order within each seed. Final epoch only; no best-seed or epoch selection.
- Train and time heads on CPU with four threads. Report accuracy, NLL, multiclass Brier, per-depth and per-label accuracy. Head latency excludes the encoder and is not end-to-end serving latency. Publish the shared feature-encoding cost separately.
- Select the strongest recurrent candidate and strongest control by mean shift accuracy. Continue only if the candidate gains at least 2 percentage points, loses at most 1 point on same-depth questions, has no worse shift NLL, wins at least two seeds, has a positive lower bound in a paired world bootstrap, and stays within three times the control's head MAC estimate.
- Bootstrap entire worlds after averaging paired training seeds, with 2,000 draws. Intervals are descriptive: they exclude seed uncertainty and selection effects. Independent confirmation is required even if the rule passes.

Official RuleTaker test members and the CLINC confirmation/calibration splits are not opened. Astra teaching has not run; these fits use the benchmark's audited gold labels.

## Reproduce

```sh
.venv/bin/python scripts/rule_memory_study.py freeze \
  --packet runs/ruletaker-depth-v1/packet --out runs/rule-memory-v1/plan.json
.venv/bin/python scripts/rule_memory_study.py prepare --device mps \
  --packet runs/ruletaker-depth-v1/packet --plan runs/rule-memory-v1/plan.json \
  --out runs/rule-memory-v1/features
.venv/bin/python scripts/rule_memory_study.py train \
  --packet runs/ruletaker-depth-v1/packet --plan runs/rule-memory-v1/plan.json \
  --features runs/rule-memory-v1/features --out runs/rule-memory-v1/fits
.venv/bin/python scripts/rule_memory_study.py report \
  --packet runs/ruletaker-depth-v1/packet --plan runs/rule-memory-v1/plan.json \
  --features runs/rule-memory-v1/features --runs runs/rule-memory-v1/fits \
  --out runs/rule-memory-v1/summary.json
```

The runner checks source, plan, packet, feature and checkpoint hashes. Existing outputs cannot be overwritten. Raw text, local checkpoints and per-question predictions stay under ignored `runs/`; public evidence contains aggregates and hashes.
