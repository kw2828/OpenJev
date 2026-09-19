# Independent capacity sizing review

**Reviewed caps: 7,200 seconds for execution and 3,600 seconds for the separate audit.** These are conservative planning choices, not statistical upper bounds, a runtime guarantee or a scientific result. The whole-tree engineering rehearsal and measured capacity support preparing a new frozen study; this review does not launch it.

## Completed evidence

The successful measurement took **719.021s** and retained 2,888 payloads / 6,485,888,085 bytes. It measured one full-shape 24-update epoch, an additional zero-update full 48-epoch constructor, nine full 64-case learned rows (all three classes and sensing panels), and five full 64-case shifted reference rows. Each row executed 50 actions per case. Models used for control were untrained synthetic tensors; the capacity fit was not deployed.

The repaired saved-output audit took **456.033s** (456.282s actual subprocess), retained 22 payloads / 843,385 bytes, and checked all 14 rows: 44,800 executed native transitions, 26,247,168 nominal candidate transitions and 9,600 selected nominal advances. Native and public-observer error maxima were zero. It performed no new model or optimizer calls. The reviewer authenticated the phase/measurement/projection receipts, their referenced timing files and current 120 profile plus 2 repair source hashes. Root separately rehashed every execution/audit payload in 3.484s; that warm-cache observation is not a throughput guarantee.

The import failure before numerical work in attempt 01 and the original attempt 02 audit's `aggregate_model_work` schema failure remain retained. The successful measurement was not rerun for the audit repair. The three actual outer process histories total **1191.948s**; nested helper timings are not added again.

## Arithmetic and allowances

The target is three 48-epoch fits (3,456 updates), 27 learned rows plus 15 references, 42 rows total, 64 cases per row and 50 actions. The exact shortened planning-horizon sum is 534; candidate work is not scaled as 50 full 12-step horizons. Training updates are scaled from 24 to 3,456 (144 times), and each measured control/audit row has three target counterparts.

| Component | Nominal seconds |
| --- | ---: |
| Training, including separately scaled full 48 setup | 882.519 |
| All 27 learned rows | 790.072 |
| All 15 reference rows | 1311.585 |
| Execution outer-overhead proxy | 12.297 |
| Audit row replay | 1326.697 |
| Saved training audit | 6.387 |
| Audit fixed/hash proxy | 13.742 |

I independently recomputed the nominal totals: **2996.472s execution / 1346.827s audit**. The measured full 48 constructor is subtracted from outer overhead before three target constructors are charged. Thus the previously identified setup duplication is absent. The audit's 144-times training scaling repeats some fixed checks and is deliberately only a proxy.

Apply a **2x planning factor** to each nominal total for a single shared-host synthetic measurement, possible thermal/load variation, fitted-state compression and unmeasured fit-to-fit variation. Then add explicit allowances:

- Execution 600s: 240s for enlarged evidence/lineage validation; 120s for copying 44 original members and six-plus-nine deployment restorations; 60s for full 48 log/order/checkpoint finalization; 120s for larger final manifests, hashes and retention; 60s for startup/terminal bookkeeping.
- Audit 300s: 120s for two enlarged evidence/lineage validations; 120s for target manifest entry/exit passes; 60s for full 48 logs, all-nine-fit lineage and report metadata.

The scientific runner validates the experiment three times and the enclosing audit twice. Every pass now reads the retained 6.49 GB capacity qualification plus approximately 0.4 GB rehearsal and original selected lineage. Those enlarged reads were not measured by the old outer-overhead proxy. Target output hashing also grows toward 19.24 GB. The allowances are explicit additional reserves, not claims that these operations were measured taking those times.

Reviewed projections are **6592.944s / 2993.653s**. The chosen caps leave **607.056s / 606.347s** beyond those already padded projections. Preserve the cooperative cap and terminal failure checks, with no automatic retry or cap increase after outcomes.

## Storage, memory and limits

Measured rows occupy 6,371,576,238 bytes. Tripling row bytes, retaining measured non-row overhead once and expanding recorded training logs from 24 to 3,456 updates gives a **19,241,362,728-byte raw proxy**. The three full training logs alone project to 12,408,336 bytes. Compression and larger final checkpoints remain uncertain. Recommend at least **100 GB available disk** before launch for raw output, metadata and later archive/scratch headroom; this is a storage reserve, not a predicted archive size.

Measured process peak RSS was 951,025,664 bytes. Post-run host context, captured 2026-09-19T05:55:21.324436+00:00, was 895,127,040,000 free disk bytes and 51,539,607,552 physical memory bytes. These were not measurements taken during execution. Full-width batch/sequence shape was exercised, but a one-epoch fit does not certify full 48-epoch peak memory. The additional six deployment models are small relative to this host; no claim about absence of other-process memory pressure is made.

The 24 update walls ranged 0.2319-0.3247s, median 0.2474s; first/last 12 means were 0.2496/0.2598s. No long-run thermal bound follows. Shift-only reference timing is extrapolated to other sensor schedules; all three learned panels were measured. No utility, loss quality, ranking or scientific efficacy informed the caps. The future actual study still requires its independently reviewed exact protocol/freeze and all-source/runtime validation.

## Bindings

- Measurement completion: `416d8b10fd8bd36758db2a03f9b62b4270cfcc079b73558b87072cf593739e13`
- Repaired audit completion: `f906d24f76af5411bca39883bdee0f3e3ce20296ad1d36e4112face655322931`
- Repaired measurement audit: `c037b4b5914f45b2252b98b9c93aff5aab81746618dc2e0237373f2b7509cd88`
- Root full-member verification: `c5d7f7e8344967a1218ff69e8c7266146f068a2e73970bd3e1047a27b01f7893`

The companion `sizing-review.json` contains the exact source/runtime maps, every measured row, all arithmetic, allowance rationale, failure/process identities and this Markdown's hash. Reviewer work was saved-file/source inspection and descriptive arithmetic only, with zero model, optimizer or native simulation calls and no changes to 116 scientific sources.
