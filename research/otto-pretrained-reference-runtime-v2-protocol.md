# OTTO reference qualification in a corrected numerical runtime

21 September 2026. This is a separate engineering qualification after the [original failed attempt and primitive diagnostic](otto-pretrained-reference-results.md). It preserves that failure and does not change any completed scientific result.

The first attempt stopped on a NumPy 2.2.6 matrix-multiplication exception before any value or policy comparison completed. A separately frozen synthetic diagnostic reproduced numerical warnings despite exact finite outputs, including before TensorFlow import. NumPy 2.5.3 returned the same four primitive answers without warnings. This is a concrete reason to test a corrected runtime, not a reason to waive value or action parity.

## Only the numerical runtime changes

Create a new `.venv-otto-reference-v2` environment. Preserve Python 3.12.13, TensorFlow 2.20.0, tf-keras 2.20.1, h5py 3.14.0 and every other installed distribution from the original environment. Change only NumPy from 2.2.6 to 2.5.3. Retain exact requirements, resolved package identity, installation and dependency-check output. The old environment remains untouched.

Before any checkpoint construction or forward call, require a CPU-only legacy-Keras import check, float32 configuration, the same fixed primitive arithmetic checks before and after TensorFlow import, and relevant synthetic tests under this actual new runtime. The preflight does not load weights, construct the released model or call an environment.

The versioned wrapper must authenticate the old runner's exact source before importing it. It may configure only the run version, expected NumPy version and required source membership. All inference functions, tensors, fixture definitions, tolerances, tie rules, budgets and complete coverage requirements remain those of the [original protocol](otto-pretrained-reference-protocol.md). The wrapper, tests, both protocols, unchanged inference sources, original failed receipt and diagnostic must be pinned in a new machine-readable plan. Commit the new plan before execution.

## Unchanged qualification requirement

Load the original HDF5 through the unchanged official model in legacy Keras and the extracted tensor archive independently through the unchanged NumPy port. Compare all eight tensors exactly by ordered identity. Evaluate all sixteen fixed raw inputs in both symmetry settings and batch sizes one, three and sixteen, including remainders: 46 paired batches and 96 paired scalar predictions.

Evaluate the same 32 physical synthetic policy fixtures and four branch-mass fixtures. Preserve all-four-action scoring, including blocked boundary moves; all original branch floors, input centering, symmetry ordering and strict first-action tie behavior remain unchanged. Require exact input/mass identity, value and cost differences at most `1e-4 + 1e-5 * abs(reference)`, and exact selected-action agreement for every fixture. No near-tie exemption or new fixture selection is allowed.

There are 82 requested TensorFlow value calls and 82 requested NumPy value calls if all work completes, plus separately recorded construction, build and weight-loading calls. Retain every comparison and append-only attempted/returned call accounting. A completed comparison with any disagreement is an unqualified result even if the process exits zero.

Run once under the existing suspend-inclusive supervisor with the same **600-second, 4 GiB RSS and 256 MiB output limits** and an exclusive output directory. Preserve failed or partial work; do not rerun this plan to obtain a pass. Require worker and supervisor closure, absent worker process group, unchanged source/input hashes and saved-output readback before reporting the result. No simulator calls, native actor construction, training or autonomous episodes are part of this qualification.

Even complete numerical and action agreement establishes compatibility only under the stated modern runtime. It does not reproduce the historical TensorFlow runtime, qualify a native/public adapter or establish control quality. A disagreement does not permit silently replacing the released policy. The original TensorFlow implementation remains available for a separately qualified native comparator.
