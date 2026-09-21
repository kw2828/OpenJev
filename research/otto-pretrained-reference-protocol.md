# Qualifying the released OTTO value model

21 September 2026. This engineering comparison prepares a stronger learned reference for the [action-head study](otto-action-head-protocol.md). It does not change that study, run new search episodes, train a model or establish a novel architecture advantage.

## Identity and runtime

Use only the authenticated official `zoo_model_2_3_2` at original OTTO revision `1467029f399dc5eeac8652499a9c8326ecab4575`. The [retrieval](../output/otto-pretrained-reference-v1/retrieval-01.json) and [extraction](../output/otto-pretrained-reference-v1/extraction-01/receipt.json) bind the original HDF5, its configuration and all eight finite float32 tensors. They total 13,390,849 parameters. The original configuration is inspected without executing its pickle.

Use an isolated Python 3.12.13 environment with TensorFlow 2.20.0, tf-keras 2.20.1, NumPy 2.2.6 and h5py 3.14.0. Enable legacy Keras before import, expose CPU only and use one numerical thread. The complete package list is retained with setup receipts. The initial setup's irrelevant parent-project discovery failure and replacement of yanked tf-keras 2.20.0 are preserved; neither attempt loaded a model. The scientific run's existing runtime remains unchanged.

Qualification compares the published inference source under this modern runtime. It does not reproduce the authors' historical TensorFlow runtime or prove identity with the separately released benchmark archive checkpoint. The HDF5 itself records Keras 2.4.0. Retain both [original-zoo](../third_party/otto/LICENSE-zoo) and [benchmark](../third_party/otto/LICENSE) notices.

## Independent loading and numerical comparisons

Freeze a machine-readable plan with source, test, runtime and input hashes before the first checkpoint forward call. Load the original HDF5 through legacy Keras into the unchanged published `ValueModel`. Independently load the extracted NPZ into the NumPy implementation. Match all four kernels and four biases byte-for-byte by layer order and weight identity, including the two same-shaped hidden kernels. A matching shape alone is insufficient.

Evaluate sixteen fixed centered arrays covering zero mass, subnormalized mass, sparse beliefs, asymmetric dense beliefs and square symmetries. Evaluate both settings of symmetry averaging in batches of one, three and sixteen, including the final remainder. Every value must satisfy `abs(actual-reference) <= 1e-4 + 1e-5*abs(reference)`. Retain each prediction, not just the maximum difference.

Compare the complete policy algebra on **32 physical synthetic fixtures**: two previously qualified sensing kernels, four agent positions (center, corner, edge and opposite corner), and four deterministic belief patterns. Add **four mechanical fixtures** that isolate branch masses zero, `0.5e-10`, `1e-10` and `2e-10`. Mechanical fixtures are explicitly separate from physical task performance. No fixture is an observed search outcome or selected for its model score.

The unchanged published `RLPolicy` receives a deterministic fixture environment. Its model wrapper records actual inputs and branch masses from that exact source call. Compare these with the NumPy adapter and retain both arrays. This checks policy algebra with synthetic helpers, not integration with the native simulator or public actor lifecycle.

Compare all four expected action costs with the same fixed numerical tolerance. Require **exact selected-action agreement on every fixture**, using the unchanged strictly-less-than-`1e-10` first-action tie rule. Report score margins and disagreements. A near tie does not exempt a disagreement or permit changing the tolerance after observing results.

## Semantics that must be preserved

- Zero-pad the 53x53 belief to a 105x105 grid centered on the agent. Do not crop, wrap or renormalize it.
- Apply three 1,024-unit ReLU layers, then the raw scalar linear output. Nonnegative final-layer training constraints are not an inference output clamp.
- Preserve the published order of eight square transforms and their float32 averaging.
- Floor each successor branch mass at `1e-10` before dividing by it. Preserve zero or subnormalized successor arrays and do not renormalize branch weights.
- Score all four actions, including boundary moves that stay in place. Do not silently insert the existing public controller's legal-action mask.

These semantics follow the [official value model](https://github.com/C0PEP0D/otto/blob/1467029f399dc5eeac8652499a9c8326ecab4575/otto/classes/valuemodel.py), [policy](https://github.com/C0PEP0D/otto/blob/1467029f399dc5eeac8652499a9c8326ecab4575/otto/classes/rlpolicy.py) and [centering code](https://github.com/C0PEP0D/otto/blob/1467029f399dc5eeac8652499a9c8326ecab4575/otto/classes/sourcetracking.py). The pinned benchmark fork preserves their 2D inference behavior.

## Closure and next decision

Run once in an exclusive output directory under a suspend-inclusive supervisor, bounded at **600 seconds, 4 GiB RSS and 256 MiB output**, with zero simulator steps or training updates. Preserve original failures, incomplete work and terminal status. Bind all inputs and source again at closure. Do not extend the run or rerun it to obtain a passing numerical result.

Report independent weight identity, centering/input identity, value parity and complete action parity as separate results. A value-only pass does not qualify the policy port. Even a complete pass still requires a separate native/public-adapter integration check and a frozen autonomous cohort before reporting control quality. If action parity fails, retain the original TensorFlow implementation as a possible comparator rather than claiming that the NumPy policy is equivalent.
