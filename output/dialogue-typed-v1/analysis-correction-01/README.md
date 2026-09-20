# Saved-reader schema correction

The first saved-only report failed before scoring because it compared the complete split-projection dictionary with an expected dictionary containing only `fit` and `evaluation`. The authenticated projection also contains four primary/secondary subsets. The failure remains at `../analysis-01/failed.json`; its recorded duration was 1.168162 seconds. All twelve training fits completed once and remain unchanged.

The separate `scripts/report_dialogue_typed_v2.py` validates all six ordered fields. It independently derives the primary held-out-service subset, the non-held-out subset, and that subset's services present versus absent from fit. It rejects missing/extra fields and corrupted subsets, including the legitimate empty absent-from-fit subset. No metadata is deleted or relaxed.

Only the original `authenticate_run` and `execute` functions changed. Authentication replaces the invalid two-key equality; execution binds and discloses this correction. All other original function ASTs, including every metric, group, threshold, continuation decision and report table function, are identical. The adjacent receipt records hashes for each unchanged function, both reporter sources, tests, the original failed report and the unchanged training completion. All 25 frozen scientific sources were rechecked unchanged.

The corrected copy passed 35 synthetic tests, including the full fake execution tree, every subpanel corruption, empty-subset acceptance and AST identity. Ruff is clean. Actual completed receipt/configuration/work schemas were inspected without loading prediction arrays. No further schema mismatch was found. The correction adds no training, inference, relabeling, probability repair or changed criterion.

Status at preparation: corrected reporter not executed. Root will publish the correction and preserved failure before a separately authorized saved-only `analysis-02` invocation. This note and receipt document that pre-execution state and are not a quality result.
