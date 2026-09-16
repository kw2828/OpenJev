# DecisionTics Doom v0.1 model card

## Intended use

Local education and experimentation with finite-action neural policies in ViZDoom. Independent work inspired by the typed-decision interface of Jev. Not affiliated with TypeSafe AI.

## Model

- Fully connected network: 11 inputs, 64 ReLU units, 64 ReLU units, 5 outputs.
- 5,253 trainable parameters.
- Shared hidden layers; 3-way steering and 2-way firing softmax heads.
- Numerical structured input; no pretrained model, text encoder, tokenizer, language decoder, or pixel encoder.
- NumPy inference; NPZ checkpoint loaded with `allow_pickle=False`.
- Feature order and checkpoint hash: [training metadata](../src/openjev/weights/doom.json).

## Training

Supervised behavioral cloning from `teacher_action`, a public, deterministic aiming-and-shooting rule. 60,000 independently generated synthetic states, seed 7. States vary visibility, aim error, target width, distance, health, ammunition, scenario type, and directive. Aiming boundaries are oversampled. Adam, learning rate 0.003, 40 epochs, batches of 512, sum of two cross-entropies. CPU training with two PyTorch threads and deterministic algorithms.

There are no human transcripts, private data, Jev outputs, scraped training corpora, pretrained weights, or reinforcement learning rewards in training. This does not implement RLCD. Full training is reproducible with the optional `train` dependency. Checkpoint hashes can vary across dependency/platform combinations even with fixed seeds.

## Validation

A separate synthetic set of 10,000 states, seed 8, measures agreement with the same teacher. Steering agreement: 99.72%; firing agreement: 99.68%. This tests imitation on the synthetic distribution, not game competence or real-world calibration. The binary Brier score is also against synthetic teacher labels only.

Real gameplay development used seeds 42-44. The shipped model was then evaluated on seeds 1000-1019, with the same seeds for rules and random controls. No checkpoint selection or training update used those gameplay results. Runs were repeated to verify telemetry changes, without changing weights or controller behavior. The complete per-episode records are in [defend-center.json](../evidence/defend-center.json).

The measured result is specific to the ViZDoom version and local platform recorded in that file. Engine/platform differences may change precise trajectories. There is no statistical claim of equivalence between the student and teacher. The tests require only basic game competence, valid outputs, and enforced constraints.

## Observation and action limits

The model receives privileged engine-generated information about visible enemies. The harness selects the target nearest the crosshair and normalizes its bounding box. It supplies distance, ammo, health, scenario category, and a fixed directive. No hidden actor list or map planning is used. Enemy classes are explicitly allowlisted; new actor classes may be ignored.

Only left, hold, right, and fire are available. In Basic, left/right mean strafe. In defense modes, they mean turn. There is no movement through full levels, inventory planning, free-form instruction following, or audio reasoning. Inference can make incorrect decisions despite returning valid types. Pacifist and empty-ammo firing restrictions are enforced outside the model.

## Limitations and likely failures

The model is bounded by its teacher: it can waste shots, switch targets poorly, oscillate around aim thresholds, and run out of ammunition. Synthetic state combinations differ from the actual game distribution. Softmax probabilities are not calibrated success estimates. Terminal death is an ordinary result, not hidden by the recording or evaluation. Full campaigns and visual-only inputs require a different observation/training pipeline.

## License

Project code and weights: MIT. ViZDoom and Freedoom remain under their upstream licenses. The NPZ checkpoint contains only this project's learned numerical parameters.
