Full public memory succeeded on 195/256 layout/order pairs (76.2%), versus 76/256 (29.7%) for last-32 history. The comparison retains all 64 layouts, four paired priority orders and five controllers. All 1,280 episodes and 103,069 actions passed exact native replay and declared-controller reconstruction.

The fixed qualification FAILED, with 16/17 checks passing: full memory missed the required 80% success rate. Every relative memory-effect check passed. These are rule-based controllers, not a trained recurrent model or an architectural result. The recipe is closed without threshold, case, window or budget changes.

[Results, chart and recorded GIF](https://github.com/kw2828/OpenJev/blob/main/research/mystery-path-memory-qualification.md).

Source, protocol and inputs were published before execution at 540ec3071ba5112be150c57aad6425f97ab0a0f6. This release includes all raw episode frames/actions, inputs, source, tests, runtime pins, replay records and reporting evidence. Each archive member is independently hashed in members.json. No model weights are included.
