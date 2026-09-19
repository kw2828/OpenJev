# Two-observation stream component: independent static review

Reviewed 2026-09-19T04:37:56.382073+00:00.

**No material blocker found in the reviewed component.** This is a source-only review of an additive prospective helper, not historical artifact authentication, scientific freshness certification, or authorization to run. No tests were rerun, numerical seeds derived, generators instantiated, production artifacts read, or model/native calls made by this reviewer.

## Exact reviewed inputs

- `src/openjev/research/reacher_two_observation_streams.py`: `03c2e600a2e6c18afe6fbd03840ef4d5cfd881f5634406492925b890d569cb69`
- `tests/test_reacher_two_observation_streams.py`: `d1207bdf2332a55ea2b58e52b4d7f31e43c32fbbcbb9ae3b73a110c71c1bcf66`
- Supporting source read: the symbolic role/settings definitions in `reacher_two_observation_protocol.py`, historical role membership in `reacher_geometry_memory_protocol.py`, and the pure manifest expansion in `reacher_random_streams.py`. Known literal410 call sites in the new model/trainer/controller tests were inspected, without execution.

The author reports 48 synthetic tests passed in 3.33 seconds and Ruff clean. This review does not independently reproduce that execution claim.

## Findings

The full role count is correct by direct source arithmetic: three per-case public/native roles plus one particle-filter parent give 4 x 64 roots, fifty decisions with five innovation roles give 250, and uniform/bootstrap give 2, totaling 508 roots. Replacing each of 64 filter parent roles with its three explicit child generators gives 636 generator identities, not 636 additional children. The reused manifest records all initial/process-noise/resample spawn keys and initial PCG64 state hashes. Both root-seed and actual-generator-state collisions are checked within new streams and against the supplied older NumPy, geometry-memory and literal NumPy inventories.

All five geometry-memory namespaces are mandatory, in fixed order, with the complete 508-role full64 membership even for smaller engineering fixtures. Their supplied manifests are recomputed from their supplied registries, including filter children. Production preparation derives its current scored namespace and excludes all four new engineering namespaces at full64 scope. Engineering validation skips scored derivation explicitly and excludes the other engineering namespaces; it does not pretend to have certified prospective scored freshness. The tests guard every named derivation against scored/historical namespace allocation.

`validate_stream_contract` binds the complete finite JSON settings and contract, returning a frozen dataclass with a private copied read-only registry. The inexpensive seed/schedule/input helpers recheck the complete settings digest and require that validated handle. Mutation of the input JSON cannot alter the handle. This is correctly described as an in-process misuse guard, not a security boundary. Schedule phase is paired by named role, the returned schedule is backed by immutable bytes, and all five full-horizon innovation arrays are generated explicitly, including unused random-extra/anchor/tail draws. No frozen protocol globals are replaced.

Literal410 sources are named and hash-attributed, including overwritten constructors, model tests, original-order tests and controller innovations. Historical Torch manifests retain their explicit seed32 convention and possible intentional aliases. There are no new Torch namespace streams, so these are reuse/provenance records rather than a claim that independent new Torch samples were obtained. Construction and order-validation draws are described honestly instead of being called zero-draw work.

## Enclosing-runner obligations

The helper validates the **structure and separation of the supplied historical inventory**. It does not read the cited files, prove that their hashes are authentic, infer their actual named seeds, or establish that the older NumPy/Torch inventories and literal call ledgers are exhaustive. Mandatory namespace names and well-formed digests alone do not establish historical truth. The source and contract say this explicitly, keep `execution_authorized` and `production_freshness_certified` false, and require caller authentication at entry and exit. The future runner must bind the actual completed lineage, all older inherited inventories and any additional literal calls before generating study inputs. No component-only test closes that obligation.
