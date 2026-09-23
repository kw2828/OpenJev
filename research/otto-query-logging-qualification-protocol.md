# Prospective durable-logging qualification

Freeze the qualification program, both producer sources, new journal module and
its tests before timing. This is an engineering comparison on fabricated events,
with no model, native environment, sampler, arrays or scientific outcomes.

Use the unchanged native Python runtime, one numerical thread, six paired blocks
of 4,096 prebuilt events per implementation and fourteen fixed files in each new
arm directory. Alternate old/new then new/old order across blocks. The old arm
uses the frozen V1 Ledger.emit and its original sampler-record size check. The
new arm uses V2 Ledger.emit with the persistent-stream journal and its integrated
record-size check. Both retain a durable flush and fsync for every append.

Construct deterministic varied attempt/return events without reading old run
data. Exclude trace construction and directory creation from timed intervals.
Include logger initialization, the complete append path, final stream close and
directory reconciliation in the total; report those three times separately. Preserve every block,
individual fsync count, complete journal and source hash. No warmup or rerun.

Admission requires prior failure-injection tests to pass, byte-identical old/new
journals in every block, exactly 4,096 durable append acknowledgments and fsyncs
before closure in each arm, no pending/poisoned state, a median paired total-time
speedup of at least 2.0, and faster new logging in at least five of six blocks.
One exclusive attempt has 120 seconds, 1 GiB RSS and 64 MiB output, including
authentication, all measured work, verification and failure evidence. Failure or
cap exhaustion closes this qualification allocation; retain partial records.

The speed comparison qualifies this fabricated workload only. It cannot prove
the new 72-path scientific cohort will finish within 900 seconds. A separate
prospective scientific protocol must retain all labels, the same horizon and
continuation criteria, entirely fresh streams, and the original incomplete V1
attempt. Neither throughput admission nor test success establishes learning or
architecture efficacy.
