# FSM author evidence publication correction

On 2026-09-25, a metadata and source review found that the earlier BLA evidence
archive included upstream example measurement files from the vendored author
repository. This is a publication-scope error. It requires correcting any broad
claim that the reserved files were never opened or copied.

The original [public evidence manifest](fsm-author-bla-results/evidence-manifest.json)
lists these members under
`output/fsm-author-engineering-v1/vendor/freq-statespace/examples/fine_steering_mirror/`:

| Member | Recorded bytes |
| --- | ---: |
| `u_300mV_train.npy` | 2,359,424 |
| `y_300mV_train.npy` | 2,359,424 |
| `u_300mV_test.npy` | 1,179,776 |
| `y_300mV_test.npy` | 1,179,776 |

The four entries total 7,078,400 bytes. These are the manifest's recorded sizes,
not new inspection of the array payloads or headers. The manifest also includes
the upstream example notebook. Its contents were not inspected for this review.

The [original packager](../scripts/package_fsm_author_bla.py) recursively included
the author engineering directory. Its exclusions covered runtimes, caches and
the original `combined_data.npz` archive, but not the vendor's `examples` tree.
It then read, hashed, archived and byte-verified every included file. The source
reviewed here matches both the manifest and the original receipt's script hash.
This establishes the byte-copy route; it does not establish numerical use of the
example arrays in model fitting or scoring.

## Evidence and limits

The [BLA admission record](fsm-author-bla-results/admission.json) names exactly
`u_100mV_train`, `y_100mV_train`, `u_200mV_train` and `y_200mV_train` as decoded
measurement members. The [restricted reader](../src/openjev/research/fsm_data.py)
selects those four members, with realizations 0..2 used for FIT and 3..5 for
exposed DEV. It reads and hashes the complete archive bytes and checks the NPZ
member-name directory; it does not numerically decode the other members.
The [BLA producer](fsm_author/scripts/fit_bla.py) calls that reader, and the
[independent auditor](../scripts/audit_fsm_author_bla.py) checks the recorded
member and partition roster before replaying retained requests. Its saved
[audit](fsm-author-bla-results/audit.json) explicitly scopes its work to saved
arrays and causal replay, not a new raw-measurement decode.

These documented study operations support the narrower numerical-use statement.
They are not a universal audit of every historical file access. Opaque copying
is not numerical consumption, but the package did access and redistribute bytes
outside the intended publication boundary. The original numerical audit is not
a certification that the publication archive contained no reserved files. This
finding alone neither shows training contamination nor proves its absence
across all historical work.

The original [publication receipt](fsm-author-bla-results/evidence-receipt.json)
identifies the archive as SHA256
`5998f0b12f8f67ded10f2695c3e710d78192d0e3d4c5d0550852dc6298356d1a`.
The manifest SHA256 is
`9b7342be784187c17671996896022e12f2749d7746ca94472c675c1fbee98cec`;
the packager source SHA256 is
`fa13dd5295943fd5932ba82969ec8c2ab9f8cc61cd976e2ff1774eb7a525e639`.
The archive hash is quoted from the existing receipt, not recomputed here.

The new NL-LFR evidence bundle omits the entire upstream `examples` tree,
including arrays and notebooks. The revised
[packager](../scripts/package_fsm_author_nllfr.py) prunes the vendor directory
before recursive payload enumeration and admits only its explicit source,
license, README and project configuration paths. It also rejects reserved
measurement filenames before payload hashing. Packaging subsequently completed:
the [new 2,026-file manifest](fsm-author-nllfr-results/evidence-manifest.json)
contains no vendor example paths or reserved measurement filenames. The
[new receipt](fsm-author-nllfr-results/evidence-receipt.json) identifies archive
SHA256 `a8b758b58c9671e0a4c581a123b1cb5c7bce680bbad9e3976cc92bbb5e44226f`.
The archived copy of this note predates this completion update; its prospective
wording is retained as part of that byte-bound snapshot.
Original scientific and publication receipts remain unchanged. This correction
does not retroactively sanitize the earlier release or declare a fresh reserve.
No example array, reserved member header, model weight or live evaluation output
was opened by the metadata/source review that established this correction.
