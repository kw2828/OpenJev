# OTTO attribution

The sampling adapters in `src/openjev/research/otto_public.py` adapt the initial-hit, source-location and hit-selection routines from [OTTO-benchmark](https://github.com/auroreloisy/otto-benchmark), commit `a6aaef6507cffd2aff79291c1019f506f616bbef`.

Copyright (c) 2023 by Aurore Loisy. The original [MIT license](LICENSE) is retained here. The upstream simulator and heuristic policies are imported from a pinned checkout and are not vendored in this directory.

The NumPy value/policy adapter in `src/openjev/research/otto_pretrained_value.py` also adapts inference formulas from the [original OTTO repository](https://github.com/C0PEP0D/otto/tree/1467029f399dc5eeac8652499a9c8326ecab4575), revision `1467029f399dc5eeac8652499a9c8326ecab4575`. Its released `zoo_model_2_3_2` weights are retrieved and authenticated separately. Copyright (c) 2022 by Christophe Eloy and Aurore Loisy. The original [zoo MIT license](LICENSE-zoo) is retained unchanged. The benchmark fork preserves the same two-dimensional value/policy inference semantics.
