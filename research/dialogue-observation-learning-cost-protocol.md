# Cost qualification for complete observation learning

Prospective protocol, 20 September 2026. This follows the complete corrected
[input preparation](dialogue-observation-learning-preparation-correction.md)
and precedes any scientific fit. It checks the effective-batch loss path,
metadata-selected extremes and checkpoint costs. It neither scores task quality
nor automatically admits the twelve-fit campaign.

## Inputs and fixed cases

Authenticate preparation completion
`d1461a1ea64b23338b2112d581479798c6618c8ccebfbde24ce131059474cf83`
and plan `4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e`,
all local manifested files, all 39 inherited sources and the original feature
cache. All 4,380 actor streams and the 20-epoch orders already exist. Select
cases by metadata alone and publish the complete cost plan before execution.

1. Deduplicate the largest effective training batches named in the six work
   maxima in `effective-batches.json`, retaining seed/epoch/start identity.
   Sort by that identity. This gives three distinct 32-dialogue batches.
2. Within TRAIN, select the first dialogue in ID order attaining each maximum
   of padded attention positions, encoder sequences, maximum chunk length with
   special tokens, padded candidate positions, real question updates and public
   USER turns. Deduplicate and sort IDs. This gives three single-dialogue cases.
3. Include the one-dialogue tail of seed 6901, epoch zero, separately.
4. Apply step 2 independently to DEV, giving three evaluation-only cases.

For every case run all four prepared arms in their declared order. Use three
updates per full batch, one warm and two measured. Use one update per training
extreme and tail. Thus the pilot has **52 optimizer updates and 1,168 training
dialogue visits**. DEV cases each have one warm and one measured forward, for
**24 evaluation forwards**. Repeated visits are not independent examples.
There are 40 case/arm cells and 412 pre-update parity dialogue encodings, hence
1,604 total dialogue encodings before expanding them into encoder batches.

## Exact computation

Every case/arm starts from the same pinned MiniLM and fresh scalar seed 6901.
Record full initialization tensor hashes. Keep eager float32 MiniLM on MPS,
float32 monitored scalar memory on CPU, one CPU thread and encoder dropout off.
Use the prepared original or number-normalized lexical cache as declared by
the arm. Encoder strings, token IDs, query inventory and candidates stay fixed.

Encode every dialogue and its schemas freshly. Backward each microbatch
immediately, retaining its full within-dialogue recurrence, then discard that
graph. Take one AdamW step per effective batch: memory rate .001, trainable
encoder rate .00002, weight decay .0001 and global clipping at 1. Compute one
shared denominator from all scored endpoints in the actual effective batch.
Each microbatch contributes its weighted endpoint sum divided by that count.
The one-dialogue tail uses its actual count. Do not average dialogue means.

Use prepared scored positions only to reproduce the loss geometry. Replace
target index with `(time + query_position + offset) % candidate_count` and
stratum with `(time + query_position + offset) % 3`. Use fixed historical
stratum weights. Set `offset = update_index * case_dialogue_count + microbatch_index`,
starting indices at zero. Actual labels, transition bins and correctness never select
synthetic targets or score the pilot. Do not save task predictions or quality
metrics. State updates still consume every public USER exchange independently
of scored positions; initialize NONE once and never reset to gold or detach
within a dialogue.

Before updates or evaluation in each case/arm, compare all selected context,
query and candidate vectors against authenticated original features. Require
maximum absolute error at most 2e-5. Count every parity encoder pass in paid
work. Require finite nonzero memory gradients, finite nonzero trainable word
embedding and first attention-query gradients, finite optimizer parameters and
absent frozen encoder gradients. The frozen encoder's final digest must equal
its initial digest; training must change the trainable encoder's digest.

Monitor actual incoming, feature and outgoing categorical beliefs and released
mass with the unchanged V2 tolerance 2e-6. Record completed and partial work,
all warm and measured times, encoder batch counts, token work and state checks.
Evaluation cases are forward-only and perform no optimizer update.

After the first full-batch case for each arm, time one checkpoint serialization
and hashing operation. Frozen arms save the scalar; trainable arms save scalar
and encoder tensors. Retain those four qualification checkpoints locally with
hashes and byte counts. They are synthetic-loss artifacts, not trained task
models. No checkpoint from this pilot initializes scientific training.

## Limits and reporting

Use an exclusive output directory. The complete model phase has a **300-second
wall cap**, **8-GiB process RSS cap**, **8-GiB sampled MPS driver-allocation cap**
and **256-MiB output cap**. Count source/input authentication, imports, loading,
transfer, monitored recurrence, backward, optimizer, synchronization, parity,
checkpoint I/O and closing. Device allocation samples do not establish a true
transient peak and are not additive to process RSS. Use the existing 300-second
parent process-group watchdog and record actual exit and cleanup. Coordinate
the shared numerical allocation before launch.

Freeze implementation, tests, this protocol and the complete selected-case plan
before the single model execution. Retain partial output and failure receipts;
do not resume, retry, trim cases, change devices or extend the frozen attempt.
Saved-output auditing may authenticate and recompute accounting but executes
no model. Report every case and all paid work even if a later case fails.

A pass establishes this tested effective-batch route and observed costs on the
fixed workload extremes. It does not prove a universal worst-case memory bound
or full-study duration. Any training-time estimate must use the complete workload
inventory, retain explicit overhead/headroom and remain labeled a heuristic.
Fix a separate whole-study cap and final scientific reporter before launching
the twelve fits. Official TEST remains sealed and no new benchmark accuracy or
novel-architecture claim follows from this qualification.
