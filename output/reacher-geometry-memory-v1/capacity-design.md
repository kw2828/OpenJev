# Engineering capacity design, not a scientific protocol

Status: proposed, no capacity execution or scored draws performed. Keep the current 90 scientific files unchanged. This design sizes the complete 12-model, 51-row experiment and its independent audit. It does not inspect or calculate effectiveness. The additive `capacity_sizing.py` is a standard-library-only arithmetic and authenticated timing reader, not a benchmark launcher.

## What has already been measured

The complete one-case engineering rehearsal used tiny inherited models, all 12 restorations, all 51 rows and the real H12/block3 CEM256 search. Execution took 66.222493s and independent audit took 32.664569s. Its plan SHA is `5fdfe0061777e60337c7b4c17780f30c062a7172637ebbac8a2f6ed4dbd4931d`; audit receipt SHA is `4793f27b9839239abd48c063dfe0b56a2f63581fed9d14cb09ece7cab84779ee`; execution completion SHA is `eda53574247047acaebba93974b9896bb486a0fa22ff7e829045ae3207302285`. Files are under `output/reacher-geometry-memory-rehearsal-v1/attempt-01`.

This establishes complete engineering wiring, not full-batch capacity. Multiplying the whole audit by 64 gives 2,090.53s, a sensitivity check that overcounts fixed overhead. The prior 300s audit cap is therefore not a supported default. The previous geometry capacity helper, `output/reacher-geometry-capacity-v1/capacity_probe.py`, supplies useful receipt/failure machinery, but its RS64 references do not measure this new native CEM256 workload. Do not reuse its timing projection or its production-checkpoint model source. Preserve its failed `profile.py` import-shadowing attempt; never name the new helper `profile.py`.

## Exact work to cover

At 50 real decisions, the terminal-shortened horizon is 12 for 39 roots, then 11 through 1 once each. The horizon sum is 534. All numbers below include all three panels and all declared fits or references.

| Work | Count |
|---|---:|
| Learned rows / physics rows / floor rows | 36 / 9 / 6 |
| Learned candidate sequences | 29,491,200 |
| Learned candidate transitions | 314,966,016 |
| Learned selected one-action advances | 115,200 |
| Physics candidate sequences | 7,372,800 |
| Physics candidate nominal transitions | 78,741,504 |
| Physics selected nominal advances | 28,800 |
| Physics candidate integration substeps | 157,483,008 |
| Physics selected integration substeps | 57,600 |
| Executed native control transitions | 163,200 |
| Public-kinematic observer nominal transitions | 9,408 |
| Particle observer nominal transitions | 301,056 |

The independent audit replays every 78,770,304 candidate-plus-selected nominal transition, separately from executed native trajectories and observer reconstruction. It must not extrapolate a cheaper vectorized scorer as a proxy for this Python/native replay loop. All selected root re-advances, resetting/forward calls, postconstraint calls, expected-action-cost arithmetic, clipping, serialization and hash passes count.

## Smallest useful next profile

Use exclusive `output/reacher-geometry-memory-capacity-v1/attempt-01/{execution,audit}` directories with fixed 600s execution and 600s audit caps. These are proposed engineering caps only, not the future scientific caps. A timeout is a preserved failed attempt, not permission to omit a class, horizon, selected action or audit. Explicitly authorize that helper before execution.

1. **Bind inputs before any draw.** Capture the current 90 source hashes, helper hash, native/runtime identity, hardware, Torch threads 2, filesystem free bytes and completed engineering-rehearsal identities. Use only `reacher-geometry-memory-engineering-capacity-v1` and its existing `control/reset/{i}`, `control/actuator_noise/{i}`, `control/sensor_schedule/{i}`, `planner/control/{t}/...`, `planner/particle_filter/{i}` and floor roles. Call the existing authenticated prior-registry contract; do not invent extra role names. Keep the production namespace excluded. Attribute any constructors to already excluded literal 410, isolate/restore the outer Torch RNG, and record initial/final RNG bytes. Reusing a named bank across classes is deliberate pairing, not a new draw.
2. **Use full-width synthetic classes.** Construct the exact four classes at GRU width 64 and MLP width 107 with isolated seed 410. Save the untrained tensors and their hashes, then set the four-value observation-head bias to `[1,1,0,0]` deterministically in every class to keep angle pairs away from zero. Preserve all dense random weights and every head. Record that explicit edit and before/after hashes. No fit, optimizer, production checkpoint or teacher is used. Fail on any invalid rollout; do not try another seed. Tiny inherited models are suitable only for schema/context checks, not learned throughput projection. Synthetic predictions can compress differently from fitted predictions, which is a stated limitation.
3. **Measure full-batch learned decisions.** Run the exact controller's CEM256 search and separately selected action at roots 0,12,32,44,49 for each of the four full-width classes and each of the three panels: 60 full-batch probes, each with 64 cases. Reconstruct every root from the saved public prefix and previously issued zero commands, with exactly one action advance between real assimilations. Do not use a repeated-assimilation shortcut. Save root/carried states, scoring arrays and full search traces. Charge prefix reconstruction separately and record each outer decision time through compression and file writes. This tests real batched operations and panel-specific public-memory semantics; it does not measure a whole learned closed-loop row. Do not calculate effectiveness summaries, method comparisons or scientific continuation checks. CEM candidate ranking and action selection remain part of the measured controller.
4. **Measure five full-batch physics decisions.** Collect one 64-case zero-action capacity episode source with the same explicit role bindings. At its saved roots 0,12,32,44,49, run the actual physics CEM256 adapter at horizons 12,12,12,6,1, preserving all 4 candidate banks and the separately paid selected advance. This is 704,512 candidate nominal transitions plus 320 selected transitions, under 1% of the full nominal workload. These are diverse native initial states with real small action noise, not 64 copies of one root. Native source collection is charged separately. Save complete journals, search traces and input identities, including failure scratch states. No privileged state enters the synthetic learned controller.
5. **Audit the recorded profile in a separate process.** Authenticate completed profile receipt and every file before using it. Use the actual independent `audit_physics_trace` plus saved-score CEM reconstruction on all five complete decisions; it must replay 704,832 transitions and selected/native/component/count checks. Audit all 60 learned probes using saved-score arithmetic, CEM and root/carried work checks, with no neural call. Replay the shared native source separately. The profile audit must retain its narrower scope: it is not the scientific 51-row audit and cannot claim a new scientific pass. Capture reading/decompression, check time and receipt hashing separately. Reuse the complete engineering rehearsal for fixed lineage/restoration/observer wiring costs.

The new, unexecuted `capacity_probe.py` uses deferred scientific imports and requires an explicit `--execute-engineering` flag. It must not call scientific `prepare`, `run_validated`, qualifications, comparisons or renderers. `capacity_sizing.py` deliberately cannot run it. Both helpers remain outside the scientific source cohort; the executable helper binds its own hash in its engineering receipts.

## Projection and cap decision

For each class and panel, interpolate decision cost from its three H12 probes and its H6/H1 probes using the full horizon histogram; multiply by three inherited fit pairs, then sum all twelve class/panel combinations. Add scaled prefix reconstruction as a separate conservative allowance for real assimilation/state carrying. This allowance double counts some action-advance work already charged by the selected probes. The synthetic profile cannot identify checkpoint-dependent compression or hidden-state effects. For physics, use the median of the three H12 decisions and the measured H6/H1 times to interpolate missing horizons, then multiply the 50-root horizon histogram by 9. Also report the slowest of all five measured decision times multiplied by 450 as a conservative planning projection. This is not a guaranteed runtime bound. Repeat this arithmetic independently for native audit time, retaining all selected and serialization work inside each measured decision.

Add observer, native collection, per-row setup/finalization and fixed source/restoration costs from the complete rehearsal, scaled only by the work dimensions they actually use. A conservative preliminary estimate may scale its non-search reference row overhead by 64; label the overcount of fixed Python/file overhead. Hash/decompression cost scales with measured bytes and file count. Nested native, geometry and row times must not be added twice. Report process peak RSS separately from tensor payload bytes.

Choose execution and audit caps separately only after these completed measurements. A concrete rule is the next 300s boundary above twice the larger applicable point/conservative projection plus 120s of phase overhead. If that exceeds the agreed time envelope, stop before scientific freeze and report the resource blocker. Do not silently reduce cases, candidates, models, selected work or replay. Two times a projection is a planning margin, not a confidence interval or guarantee. Shared-host timings are descriptive throughput, not isolated latency or FLOPs.

## Storage and failure accounting

The helper computes a raw numeric-array subtotal from explicit dtypes and shapes. Candidate learned scoring alone is 64 bytes per predicted transition. Each physics bank uses 134 bytes per transition plus 69 bytes per sequence, including qpos/qvel endpoints, completion flags and geometry components. CEM traces, native episodes, states, metadata, inherited copies and packaging are additional. This is not a peak-RAM estimate or a compressed-file bound.

Project stored bytes from the actual 60 full-batch learned probes and five physics journals, using the same horizon weights and separately scaling trace/episode/observer files. Record both uncompressed array payload and actual NPZ/JSON bytes. Before freeze, require at least the larger of 100GiB or four times the conservative complete raw-run estimate free, plus already retained engineering files. That reserve covers the run, a packaging copy, multipart assets and failure headroom; it is a prospective operating rule, not measured disk use. A read-only check found about 912GiB currently available, but recheck immediately before execution.

Each phase writes `started.json` before expensive work. Bind source/RNG/model identities and fixed caps, append a cursor, and preserve any partial native bank, raw arrays and error in `failed.json`. Completion requires unchanged scientific source hashes, complete declared coverage and a final cap check; a late cap failure moves the provisional completion receipt aside. Do not overwrite a failed attempt or silently retry. Charge unsuccessful engineering time separately from later scientific execution and publication costs. No scientific readiness conclusion follows until both the full engineering rehearsal and this prospective capacity profile finish successfully.

## Future invocation after review

No command below has been executed. The helper writes only its exclusive engineering attempt directories.

```sh
PYTHONPATH=src:scripts .venv-robotics/bin/python output/reacher-geometry-memory-v1/capacity_probe.py run output/reacher-geometry-memory-capacity-v1/attempt-01 --execute-engineering
```

After a successful execution, independently calculate the SHA256 of `execution/completed.json`, then supply that exact value to the separate audit process:

```sh
PYTHONPATH=src:scripts .venv-robotics/bin/python output/reacher-geometry-memory-v1/capacity_probe.py audit output/reacher-geometry-memory-capacity-v1/attempt-01 --completed-sha256 <verified-completion-sha256> --execute-engineering
```

The capacity plan reuses the prospective study's namespace/exclusion template. Its discarded-constructor description concerns the future 12-checkpoint study. The actual capacity receipt instead records four synthetic seed-410 constructors and zero checkpoint restorations; do not conflate the two scopes.
