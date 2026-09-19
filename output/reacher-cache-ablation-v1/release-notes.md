Completed five-arm Reacher cache comparison: **the recurrent-superiority continuation rule failed; 15 of 28 checks passed**.

The cached MLP lowered mean native control cost by **6.32% on six-step observation gaps and 5.57% on ten-step gaps** versus persistent GRU memory, improving every paired fit on both panels. It used about **38% more whole-row controller time**. All 15 fits and 60 controller rows are retained.

Persistent recurrence predicted missing angles and one-step rewards more accurately on the common prediction cohort but controlled worse than cached MLP. Caching helped the MLP and hurt the reset GRU relative to their respective current-only controls. These development findings do not establish cache equivalence, biological superiority or a new architecture.

- [Full results and limitations](https://github.com/kw2828/OpenJev/blob/main/research/reacher-cache-ablation.md)
- [Every paired comparison](https://github.com/kw2828/OpenJev/blob/main/evidence/reacher-cache-ablation-v1/figures/paired-comparisons.png)
- [Recorded first-case schematic replay](https://github.com/kw2828/OpenJev/blob/main/evidence/reacher-cache-ablation-v1/replay/first-ordinary-case-pair0.gif)

The fixed replay shows case 0 / pair 0 and all five models. That single episode favors persistent GRU despite the aggregate failed criterion. It is a schematic drawn from saved states, not simulator camera pixels or a selected successful episode.

The saved-output audit replayed **235,200 native transitions with zero discrepancy**, without new learned-model calls. Execution took 1,234.34 seconds and the audit took 36.72 seconds. Equal data and updates do not match total computation.

The scored archive includes all checkpoints, traces, frozen sources, protocol, closed supervision, audit, figures and replay. Concatenate `.partNNN` assets in receipt order to recover the complete gzip archive. Check its SHA-256 in `receipt.json`; `manifest.json` binds every archived member. Historical lineage dependencies remain separate releases. Capacity and rehearsal archives are separate engineering evidence, not scientific results.

Packaging sidecars record their pre-upload state. The repository's final release-verification receipt records publication and verifies all remote asset sizes/digests; those immutable packaging sidecars are not rewritten.

Frozen plan SHA-256: `7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa`

Final audit receipt SHA-256: `d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790`
