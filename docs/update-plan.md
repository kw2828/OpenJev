# Decision interface update

1. Define a game-independent English context/question/candidates contract with caller-owned IDs.
2. Implement pinned local Qwen next-token candidate scoring, lazy offline loading, bounded input and serialized inference. No generated explanations or fabricated fallback scores.
3. Add an editable decision playground with examples, candidate probabilities, model status and JSON export.
4. Connect Doom through that same service with English instructions, bounded recent history, and six executable action combinations. Retain the tiny imitation model as a separately labeled baseline.
5. Verify contracts, actual local inference, instruction changes, actual Doom actions and browser operation. Publish measured development evidence and scope limitations.
6. Update README and model documentation, run checks, commit and push.

Conformal prediction, GLiClass, learned memory, latent recurrence and new RL training are future experiments. This update does not claim to implement them or reproduce proprietary Jev/RLCD.
