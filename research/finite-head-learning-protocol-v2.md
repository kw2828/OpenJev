# Separate qualification after a test-only correction

The original [head-initialization protocol](finite-head-learning-protocol.md)
stopped during qualification. Its original process closed in 7.017424292 seconds:
372 tests passed and three failed, with one warning. All three failures came
from one parameterized assertion expecting `ValueError` when the public
`blind_rollout` rejects the unsupported `oracle_prefix` argument with `TypeError`.
The model rejected the prohibited input as intended. No exposure probe,
scientific registration, scientific training or scientific evaluation started.

Preserve every original v1 source, its engineering registration and snapshot,
the failure receipt, logs and native closure. The new v2 registration binds that
evidence and verifies that the only model-test change is the expected exception
class and matching message. The corrected test lives in a separate file;
the frozen original is not edited or rerun. New phase wrappers use exclusive
v2 paths and their own source snapshot. Their tests cover the same admission
and closure behavior with the new paths.

Every scientific choice in the original protocol remains unchanged: all three
models, their initializers, the 352-parameter roster, losses, optimizer, update
counts, safety caps, generator, metrics, 15 continuation conditions and the
descriptive-only anchor. The same unused scientific namespace 436260924 and
five reserved seeds remain fixed. The engineering smoke remains 946001/946101;
the previously unstarted exposure probe remains 946201/946301. No effectiveness
measurement selects these settings. The repeated small smoke is engineering
qualification of corrected assertions, not additional scientific evidence.

This is a separate registered qualification, not a retry or reinterpretation
of the failed v1 attempt. First require the complete v2 test suite and fixed
exposure projection to pass and its original process to close cleanly. Only
then may the v2 scientific registration admit the unchanged experiment. Keep
the original 300/1,200/600-second native caps, sampled 4 GiB RSS and 512 MiB
phase-output bounds. Any further failure stops the new attempt; no edits or
reruns within its registration.

Publish both qualification outcomes and complete source snapshots alongside
any later scientific result. Correcting this assertion is not a trained-model
improvement, evidence of noise transfer or an architectural contribution.
