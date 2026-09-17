# Full-context text representation diagnostic

**Status: frozen before fitting; results pending.** This follow-up was planned after weak early results in the [sentence-memory screen](rule-memory-study.md). It reuses development worlds, so it is adaptive research rather than independent confirmation.

Instead of separately embedding each sentence, MiniLM sees the full English context and assertion as a tokenizer pair. This preserves token interactions across rules, facts and the question. Every audited example fits without truncation: maximum 252 tokens in training, 240 in same-depth development and 268 in shifted development.

Two conditions share the pretrained encoder and initial classifier:

1. **Frozen joint encoder.** Mean-pool the complete context/assertion representation and train a binary linear head for 30 epochs, batch 256, AdamW learning rate 0.001 and weight decay 0.01.
2. **Fine-tuned joint encoder.** Start from that fitted head and pretrained encoder, then update all weights for three additional epochs, batch 32, AdamW learning rate 0.00002 and weight decay 0.01. Gradient norm is capped at one. This uses more parameters and compute; it is not a cost-matched architecture win.

Both use the same 19,809 audited training questions, three seeds (17, 29, 43), final-epoch outputs and the unchanged two development parts. Source IDs, labels, proof annotations and annotated depth are kept outside text inputs. We report every seed, accuracy, NLL, Brier and accuracy by depth and label. No development-based checkpoint selection is allowed. Only the supervised training loss reads gold labels while fitting.

This tests whether stronger context representations improve a weak local baseline. It does not establish a novel architecture, teacher distillation, calibration or transfer to arbitrary candidate sets. Astra API access is still unavailable, and no teacher requests are made. Official RuleTaker test data and CLINC confirmation/calibration data remain unopened.

```sh
.venv/bin/python scripts/rule_crossencoder_study.py freeze \
  --packet runs/ruletaker-depth-v1/packet --out runs/rule-crossencoder-v1/plan.json
.venv/bin/python scripts/rule_crossencoder_study.py run --device mps \
  --packet runs/ruletaker-depth-v1/packet --plan runs/rule-crossencoder-v1/plan.json \
  --out runs/rule-crossencoder-v1/fits
.venv/bin/python scripts/rule_crossencoder_study.py report \
  --packet runs/ruletaker-depth-v1/packet --plan runs/rule-crossencoder-v1/plan.json \
  --runs runs/rule-crossencoder-v1/fits --out runs/rule-crossencoder-v1/summary.json
```

Hash-checked local evidence includes initial plans, feature arrays, checkpoints, all logits and final metrics. Only aggregate evidence is published.
