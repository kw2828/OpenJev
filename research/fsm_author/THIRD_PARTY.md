# Source and license notices

This isolated component is licensed under GPL-3.0-or-later. It integrates the
published `freq-statespace` implementation by Merijn Floren, pinned to commit
`a79e8c567b018a6c9462528fc1e10b77fd19b3e2`:
[source repository](https://github.com/merijnfloren/freq-statespace/tree/a79e8c567b018a6c9462528fc1e10b77fd19b3e2).
The upstream license and source notices remain applicable. Its vendored FSID
implementation carries GPLv3 notices in the original source.

The accompanying LICENSE reproduces the upstream GPLv3 license text unchanged.
The package metadata and upstream classifier specify GPLv3 or later. Installed
dependencies retain their individual licenses; their exact versions are locked
in `uv.lock`. This component does not change the license of unrelated OpenJev
components.

The FSM measurement data are separate CC BY 4.0 material from Merijn Floren,
KU Leuven, and Floren et al., ISMA-USD 2024. No source measurement archive or
author-trained weights are included in this component. Any later derived-data
release must retain the original data notice and document its transformations.
