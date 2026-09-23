# Exact teacher-value cache opportunity

Prospective systems diagnostic, after the score-forecast collection stopped.
The original 67/90 incomplete result, its 900-second limit and its unstarted
learning stages remain unchanged. This diagnostic does not resume that run,
fit its TRAIN labels, select replacement validation cases or evaluate its
45 forecast conditions.

The practical question is whether repeated teacher computations can be removed
while preserving all annotations and original controller behavior in a later,
separately frozen collection. Do not thin labels or change model families as
part of this diagnostic.

## Fixed scope

Use all 22,296 returned requests in the original saved journals, including the
explicitly interrupted episode's returned prefix. Preserve their order and
all 68 episode resets. The later attempted but unreturned forward is not a
cache request with a known answer. Keep complete and partial episodes separate.
Report all twelve stage/setting/collector cells, including unstarted cells.

Both diagnostics use one LRU cache of capacity 64, reset at every episode.
There is no cache-size, reset-policy or key search after reading results.

1. **Metadata precursor:** use the setting, public position and recorded
   float64 posterior SHA256 as a conservative repeated-public-state proxy.
   Numerical scores and feature values are ignored. Decode the entire gzip
   stream, including its trailer, and require all chronological sample rows.
   A low proxy count does not rule out repeated float32 model inputs.
2. **Input replay:** reconstruct the posterior from the saved public reset,
   actions and observations using the unchanged public filter and known kernel.
   Require every reconstructed posterior hash and mass to match the original
   witness, including final updates. Rebuild the sixteen original action/hit
   branches with the same float64 arithmetic and final float32 conversion.
   Key the cache by complete float32 `[16,105,105]` bytes, not a digest alone.
   Require each rebuilt float32 `[4,4]` branch-mass array to byte-match that
   request's original saved `branch_masses` array.
   Join every request to its original saved sixteen branch values. Every hit
   must equal the current saved float32 value bytes exactly.

The original journal did not record a model-input hash. Reconstruction therefore
rests on the pinned original arithmetic, fabricated geometry qualification and
all public-posterior and branch-mass witnesses. Do not describe it as a direct
comparison against recorded original input bytes.

The input replay runs regardless of the metadata precursor's hit count. It
reads historical returned branch values solely to verify cache consistency,
not for fitting, forecast evaluation or model selection. It makes zero new
TensorFlow, native-environment, optimizer or learned-model calls. Public-filter
reconstruction and branch construction are explicitly counted computations.

Cache only the sixteen model values. Recompute the current branch probabilities
and retain the original TensorFlow Q reduction in any later deployed adapter.
Different float64 posteriors can produce identical float32 model inputs but
different branch probabilities; cached four-action Q scores would therefore
be invalid. No score rounding, approximate matching or posterior repair.

## Resources, evidence and continuation

Authenticate the original plan, failed receipt, closed supervisor, verified
stop summary, all referenced payloads and relevant source/kernel hashes before
decoding. Freeze each diagnostic's source, fabricated qualification, plan and
input hashes before its single execution. Use exclusive outputs and preserve
failures without retries or extensions.

- Metadata precursor: 60 seconds, 256 MiB RSS, 32 MiB output, stdlib only.
- Input replay: 120 seconds, 1 GiB RSS, 128 MiB output, original native NumPy
  runtime, one CPU thread. No TensorFlow import or model construction.
- The 64 full input keys alone occupy 45,158,400 bytes; report actual retained
  cache bytes and whole-process peak RSS separately.

Record each request, cache hit/miss, eviction, episode and original forward
time. Report all requests and all cells. Original forward time associated with
hits is a retrospective avoided-work estimate; it is not a measured speedup,
and excludes future cache overhead. Preserve original partial-path scope.

Advance to a separately frozen native cache qualification only when the input
replay finishes with exact witness, mass and cached-value agreement for every
returned request, both settings contain at least one hit, and hits account for
at least 40% of the original recorded returned TensorFlow forward time.
All conditions are required. The threshold is an engineering headroom screen,
not a confidence bound or architecture result. Metadata proxy hits cannot
substitute for this test.

A passing diagnostic would still require measured native hit/miss parity,
unchanged actions, and cache overhead accounting before collecting fresh
learning/evaluation cases. All 90 planned paths, original horizon, full
annotations, twelve fits and the 45-condition comparison remain the intended
learning experiment. No efficacy or novelty claim is established here.
