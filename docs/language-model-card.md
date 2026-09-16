# OpenJev language backend v0.2

## Identity

- Base: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- Runtime artifact: [mlx-community/Qwen3-4B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit).
- Immutable revision: `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`.
- Execution: MLX on Apple Silicon. Exact dependencies are in `uv.lock`.
- No additional training, RL, distillation or calibration was performed.
- Qwen weights remain Apache-2.0 under their upstream license; they are downloaded separately, not committed to this repository.

## Intended use

Local experimentation with bounded English decisions. The model itself is multilingual. English-only is the UI/evaluation scope, not a claim that the weights lack other languages or that requests are language-detected and rejected.

Arbitrary question and candidate text can be submitted without intent registration. That interface flexibility does not establish reliable zero-shot task generalization. Questions requiring unavailable facts, intricate arithmetic, spatial dynamics, long histories or instruction conflict resolution may fail confidently.

## Evidence

The disclosed [development receipt](../evidence/language-development.json) includes synthetic text tasks, an exact candidate-order permutation check and a pair of Doom instructions that change only whether firing is desired. Prompts were authored to verify this implementation. They are not a representative benchmark, sealed holdout, reliability estimate or adversarial robustness study. A single prompt-injection fixture passing is not a prompt-injection defense claim.

The paired Doom check evaluates model choices before execution. The observation uses the `hunt` preset in both cases, so a no-fire choice cannot be attributed to the hard pacifist override. Full gameplay receipts, when linked in the README, measure specific scenarios and seeds only. They are not comparable to the tiny baseline's 20-seed average as an aggregate quality claim.

## Limits

- Scores are conditional next-token letter probabilities, not calibrated outcome probabilities.
- Quantization can alter logits, scores and selected actions compared with original weights.
- Sorting descriptions stabilizes exact order permutations; wording and candidate-set changes can still alter choices.
- Candidate-letter entropy is ordinary categorical entropy. The live scorer does not apply semantic grouping or conformal prediction. Separate opt-in research utilities are described in [the research plan](research.md).
- The language model runs synchronously with gameplay. It does not currently meet the tiny controller's action rate.
- Doom observations use engine-provided visible actors and variables. There is no learned pixel perception, map navigation or language-to-new-action synthesis.
- Four recent observations and selections are serialized. There is no newly trained recurrent memory or latent thinking loop.
- The existing tiny 5,253-parameter model remains a separate imitation baseline, described in its [own model card](model-card.md).

## Research extensions

GLiClass candidate scoring should first be compared with this frozen baseline on unseen questions, candidate sets and instructions. Conformal sets require a defined target and separately held calibration data, including treatment of episode dependence and policy shift. Latent recurrence requires evidence that extra compute improves difficult choices. None should be described as a demonstrated improvement before those experiments exist.

## Portable hosting checkpoint

The Linux container uses a different, smaller Qwen3-0.6B checkpoint with Transformers CPU float32. Its identity, limitations and deployment instructions are in [hosting](hosting.md). The Mac 4B evidence above does not transfer to that model.
