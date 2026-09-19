# Independent public-history auditor review

**Static review clear: no material causal, schema or identity-binding issue found.** Only the new source and handwritten tests were read. No model, simulator, training, production-data or scientific-output operation was performed. No source or test was edited.

Reviewed:

- [Auditor](../../src/openjev/research/reacher_two_observation_audit.py), SHA256 `d4957a67460cb1592cd4db0cf0fda4a32c2d7b7265d2b23d1eac14bff61b2335`.
- [Tests](../../tests/test_reacher_two_observation_audit.py), SHA256 `789a468a0777c73ef33377434a7552bd0a218b93e055dcd8c63e2adfc326fcea`.

The author reports 76 synthetic NumPy tests passed and Ruff clean. This reviewer inspected the tests and verified both hashes, but did not rerun them.

The reconstruction requires exactly `step+1` actual public packets and `step` issued commands. It starts at the older of the two most recent valid observations, or the first observation before a second exists. Left padding, integer indices, actual target, binary validity and measurement age are reconstructed independently. Commands occupy the edges between the retained packets, including the interval between the two valid anchors and every subsequent missing packet.

The full-window boundary is correct: twelve packets and eleven commands are accepted; a subsequent visible observation moves the anchor before the retained suffix is formed. Every earlier boundary must also fit. A final short suffix cannot hide an unsupported overflow earlier in the episode. Terminal observation 50 permits buffer inspection, while decisions remain restricted to 0..49.

Unavailable angular placeholders are removed before finiteness checking, including NaN and infinity. Visible angles, known public fields and commands must remain finite. Static targets and ages are checked across the entire supplied prefix, not just retained rows. The stated float32 age tolerance does not relax exact equality between reconstructed evidence and saved buffers. Input arrays are not mutated; reconstructed outputs own their storage.

The root requires the exact ten-field state schema, zero imaginary depth/pending command, and exact independently reconstructed public buffers. The selected carried state must preserve all real evidence, retain the same real index, set depth to one and contain the separately supplied actually issued command. Its target and missingness remain public-clock consistent and its age advances one interval. Strict float32/bool/int64 shapes and finiteness reject malformed states, while actual-class configuration, parameter count/bytes and lowercase weight hash are compared with an external identity binding.

This helper verifies the relationship between supplied evidence and saved state. It does **not** authenticate that the public prefix was physically executed, recompute learned hidden states or predictions, certify CEM selection/geometry arithmetic, validate training or source lineage, or establish whole-run coverage. Its return value explicitly preserves those limits. The future enclosing saved-output auditor must provide authenticated native-derived prefixes, model identities and the remaining study checks. No experiment readiness or effectiveness follows from this review.
