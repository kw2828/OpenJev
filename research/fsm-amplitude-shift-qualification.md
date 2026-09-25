# Amplitude-shift reader qualification

**Reader qualified; empirical comparison unregistered and unrun.** The original
fabricated check passes **41 tests plus Ruff**, with exit 0 and unchanged source
hashes. Peer source review agrees. This supports only the restricted data and
request interface, not a model-performance result.

The [new reader](../src/openjev/research/fsm_shift_data.py) authenticates the same
opaque snapshot by caller-supplied registered SHA-256 and byte count before ZIP
or NumPy inspection. It decodes exactly `u_300mV_train` and `y_300mV_train` and
requires native float64 `[8192,3,6,2]` values. It returns twelve separate,
chronological records with immutable owned storage. Existing FIT/DEV readers
and their source hashes stay unchanged.

Fabricated poison cases verify that the other ten known member headers remain
unopened. Source mismatches reject before decoding. Other fixtures cover schema,
record identity, boundary and ownership failures. Public inputs contain C100
observed outputs, 99 aligned past inputs and H128 future applied inputs; future
output targets use a separate scorer API. Public-window construction does not
inspect future output values. No normalizer is fitted or applied by this reader.

[Original qualification receipt](../output/fsm-shift-engineering-v1/reader-qualification-01/receipt.json)
and [process observation](../output/fsm-shift-engineering-v1/reader-qualification-01/original-process-observation.json)
retain actual commands, exits, logs and source snapshots. The reader and test
SHA-256 values are `199982be6f233eda14fd38412987567851f026df8f7cf14766fc8d2c805f1bca`
and `5f73c1897bd8f1f10839c6ade88f9c88173b003680448d3ebe5d440100dd1d84`.

No real measurement archive, model, training or empirical shift evaluation was
used for this qualification. The [28-instance proposal](fsm-amplitude-shift-proposal.md)
still needs qualified inference adapters, a complete evaluator and independent
replay, fixed checkpoint identities, timing order, resource limits and published
registration before numerical access. The prior publication-copy caveat remains
in that proposal. Passing reader tests does not open the reserve.
