# Stronger learned reference and symmetry-aware readout

21 September 2026. Source review and a candidate mechanism for work after the [running action-head pilot](otto-action-head-status.md). The official reference weights and configuration have been retrieved and byte-authenticated. Static format/config inspection is complete; no reference inference has run, and no symmetry-aware model has been fitted. Neither proposal changes the current frozen run.

## A released learned comparator

The official OTTO zoo has a matching two-dimensional lambda3/R2 model, [`zoo_model_2_3_2`](https://github.com/C0PEP0D/otto/tree/1467029f399dc5eeac8652499a9c8326ecab4575/zoo/models/zoo_model_2_3_2), pinned at `1467029f399dc5eeac8652499a9c8326ecab4575`. The retrieved **53,581,916-byte weight file** and **147-byte configuration** match the official Git blob identities and the [retrieval receipt](../output/otto-pretrained-reference-v1/retrieval-01.json), preserved byte-for-byte from the local retrieval directory. Their SHA-256 hashes are `1efb73aa38e0fd8b08d6d03059c0db4664da3afb8eaff1d7ab9b363f8e7ad37d` and `ca12567f2e0333192a7a0b0c177d519bdae60749ddb8005a1456608fd6782b4c`, respectively. The repository is MIT licensed.

The independent [static inspection](../output/otto-pretrained-reference-v1/inspection-01/receipt.json) confirms HDF5 magic and printable names for four dense layers' kernels and biases. Nonexecuting pickle-opcode inspection confirms `Ndim=2`, three hidden layers of 1,024 units, zero regularization and mean-squared-error training loss. The expected eight tensor shapes come from this configuration and the published network code; HDF5 dataset shapes, dtypes and values have **not** been decoded. A later isolated HDF5 reader can inspect the Keras layer/weight metadata and extract those eight tensors without training. Numerical forward and complete policy parity still require separate qualification.

The separately released benchmark model `isotropic-53x53_drl` is bundled in the **2,099,977,090-byte** `zoo.tar.gz` under CC-BY4.0. [Official archive metadata](https://zenodo.org/api/records/7586357). Matching names/configurations do not prove the two weight sets are identical. Start with the smaller individual release as an official pretrained reference; reserve the benchmark-checkpoint label for verified weight identity.

The [benchmark value model](https://github.com/auroreloisy/otto-benchmark/blob/a6aaef6507cffd2aff79291c1019f506f616bbef/isotropic/classes/valuemodel.py) takes a **105x105 agent-centered full probability grid**, flattened to 11,025 inputs. Three 1,024-unit ReLU layers feed one scalar remaining-search-time output: **13,390,849 parameters**. The [policy](https://github.com/auroreloisy/otto-benchmark/blob/a6aaef6507cffd2aff79291c1019f506f616bbef/isotropic/classes/rlpolicy.py) evaluates sixteen action/observation successor beliefs with eight symmetry transforms per decision. It retains exact filtering and lookahead; it is not a direct compact-memory policy.

The [published benchmark](https://arxiv.org/html/2302.00706v2) provides a stronger learned-planning reference, but its reported outcomes are not measurements on our seeded cohorts. Reuse avoids training a large reference from scratch. [Upstream runtime pins](https://github.com/C0PEP0D/otto/blob/1467029f399dc5eeac8652499a9c8326ecab4575/pyproject.toml) specify TensorFlow 2.8; the [official macOS wheels](https://pypi.org/project/tensorflow/2.8.0/#files) target x86-64 and older Python versions than this Apple Silicon Python 3.12 runtime. Tensor extraction and an isolated compatible runtime or numerically qualified inference port are needed before comparison. Preserve centering, symmetry averaging, floors and tie behavior; report any semantic differences. No framework was installed or model loaded during this inspection.

## One concrete readout hypothesis

A dense absolute-grid head must learn rotated versions of a decision separately, as well as the interaction between position and global DCT coefficients. These are plausible learning burdens, not an established diagnosis of the current pilot. A candidate is one shared scalar network in action-centered orientation frames:

\[
s_a(x)=\tfrac12\sum_{g\in D_4:\,g(a)=\mathrm{north}} f_\theta(T_g x).
\]

There are two transforms for each action. Rotate that action to north, evaluate both left/right reflections through the same `input -> 32 -> 16 -> 1` head, then average. Reindexing the group gives `s[h(a)](T_h x) = s[a](x)` for every square rotation/reflection, independent of trained weights.

A DCT-II x-reflection multiplies coefficient `(u,v)` by `(-1)^u`; a y-reflection uses `(-1)^v`; transposition swaps the frequency indices. The square 16x16 truncation is closed under these operations. Thus transforming compact evidence needs sign flips and transposes, without inverse DCT or a reconstructed posterior. Transform the complete support mask, position and legal-action indicators consistently. Initial/current hits, elapsed time and known sensing length are unchanged; the centered radial initial prior is symmetric.

Apply each transform **before** the shared TRAIN standardizer. Arbitrary coordinate-specific standardization does not commute with the signed coefficient transforms. Batch all eight scalar evaluations, and charge their full computation and mask processing. A faster result is not guaranteed.

Use the identical shared architecture over all **2,809 full-belief values of 53*sqrt(p)** as the strong full-state control, with its honestly larger input-layer parameter count. Also include an ordinary same-width head trained with D4 augmentation. This separates exact architectural weight sharing from ordinary augmentation. Keep both DCT fills, recent history, equal training cases and all fitting seeds.

Before quality work, qualify all eight score permutations at boundaries and asymmetric masks, DCT transformations against independently transformed synthetic fields, anisotropic standardizers, both fills and sensing regimes, no decoding/planner calls, and absence of state mutation. Test unique-optimum action equivariance and tied-action-set equivariance separately. Deterministic first-index tie-breaking cannot be equivariant at every symmetric state.

This construction borrows established symmetry methods, including [group-equivariant networks](https://arxiv.org/abs/1602.07576) and [frame averaging](https://arxiv.org/abs/2110.03336). Symmetry alone is not a novelty claim. The unresolved empirical question is whether compact recurrent evidence retains autonomous quality at lower complete cost than an equally improved full-belief controller. Missing information, longer histories and changed-prior extrapolation remain separate issues.
