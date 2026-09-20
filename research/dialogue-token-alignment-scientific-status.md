# Token alignment: scientific campaign

**Training started on September 20, 2026 at 08:29 UTC. No quality result is available yet.**

The question is whether matching words in a dialogue to words in its candidate
answers improves actual decisions. The comparison contains three fresh fits
of each model:

| Model | What changes | Parameters |
|---|---|---:|
| Existing candidate-attention scorer | Baseline | 173,506 |
| Token-mean comparison | Shared token comparison without alignment | 124,482 |
| Token-aligned comparison | Bidirectional soft token alignment | 124,482 |

All nine fits use the same 29,211 training rows, exact paired orders, 20 epochs,
and ordinary stratum weighting. The main evaluation contains 578 changed and
7,241 retained rows from six services excluded from fitting. These are
historically exposed TRAIN development examples, with the correct previous
value supplied. This experiment is not autonomous recurrent memory.

The aligned model must improve changed accuracy by at least two percentage
points and reduce wrong-branch decisions by at least two points against both
controls. It must also satisfy every paired-seed consistency and retained-value
and rare-type harm limit. All 22 checks were fixed before fitting. No quality
report will be run until all nine fits complete and authenticate.

## Resource allocation and evidence

The earlier [cost screen](dialogue-token-alignment-capacity-results.md) remains
**NOT ADMITTED under its original 48-minute threshold**. It projected 77.27
minutes and produced no fitted checkpoint or quality result. Its original
scientific proposal remains unexecuted.

This separate campaign allocates a hard **two hours of local CPU time** before
observing any quality result. It changes the resource allocation while keeping
all data, models, seeds, epochs and scientific criteria. It has one training
attempt, no resume or time extension after failure, a 6 GiB peak-memory cap,
and a 512 MiB output cap. Other numerical work in the coordinated research task
is held during training; unrelated host load is still recorded.

- [Scientific protocol](dialogue-token-alignment-scientific-protocol.md)
- [Source and exact-order freeze](../output/dialogue-token-alignment-scientific-v1/protocol-01/completed.json)
- [Synthetic preflight](../output/dialogue-token-alignment-scientific-v1/preflight-01/summary.json)
- [Training runner](../scripts/study_dialogue_alignment.py)
- [Saved-result reader](../scripts/report_dialogue_alignment.py)
- [Independent primary-result audit](../output/dialogue-token-alignment-scientific-v1/audit-01/README.md)

Source and protocol were published at commit `cae4a66`; the metadata-only freeze
was published at `0fc677c`, before fitting. It completed in 4.59 seconds with
zero model calls, preserving 47 source files and all three original order files.

Frozen plan SHA256:
`e700080ee2dd28c83c0c13a0dad2640cdc83fb1992efb4bfd3777a90111877c1`.

This page records the launch, not successful completion. The final execution,
report and independent audit receipts will determine the outcome. No new
accuracy, calibration, biological-wiring or architectural-novelty claim is made.
