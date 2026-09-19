# Public release verification

The [complete study archive](https://github.com/kw2828/OpenJev/releases/tag/research-reacher-memory-ablation-v1) was published on 2026-09-19 at 00:34:29 UTC, targeting commit `607a7d29d9ad615df3973e7f82c30eb02e21e529`. The release contains 10 assets totaling 2,313,597,019 bytes: three scored-archive segments, two separate engineering archives and five packaging sidecars.

[release-verification.json](release-verification.json) records the final public state. All 10 GitHub upload states, byte sizes and server-reported SHA-256 digests matched the local manifest. The public `receipt.json`, `manifest.json` and `SHA256SUMS` were also downloaded and hashed. Large archives were reopened and all 4,263 scored-archive members verified locally before upload; the large remote archive segments were not downloaded again.

The adjacent [README.md](README.md), [receipt.json](receipt.json), [manifest.json](manifest.json), [packaging-started.json](packaging-started.json) and [SHA256SUMS](SHA256SUMS) are unchanged packaging artifacts captured before upload. Their statements that publication had not occurred, including `uploaded: false` where present, describe that earlier state. This note and the final release verification record the subsequent publication without rewriting those artifacts.

The scored archive exceeds GitHub's per-file limit, so its three consecutive byte segments must be concatenated in the order listed in `receipt.json`, then checked against the whole-archive SHA-256 before extraction. Engineering archives are separate and do not constitute scientific results. Historical upstream artifacts listed in the manifest may still be needed to rerun the audit.

The first draft-creation request used a short commit ID and was rejected with HTTP 422. After verifying the full remote commit and absence of a matching release, draft creation succeeded with the full ID. The single upload command completed successfully. The draft became public only after all 10 assets passed verification. The creation attempt, upload records, release notes and verifier are preserved under `output/reacher-memory-ablation-v1/`.

Publication made no new model or native-environment calls and did not alter the frozen protocol, execution, audit or 25-of-25 scientific gate. The [research report](../../../research/reacher-memory-ablation.md) explains the positive memory result and its limits, including the close feedforward baseline. Passing this study does not establish biological wiring or a novel-architecture advantage.
