# Published FSM baseline adapter

This isolated GPL-3.0-or-later component pins the authors' `freq-statespace`
implementation and provides a causal, history-only forecast interface. It is
a conventional reference for OpenJev's recurrent experiments, not a new model.

From the OpenJev repository root:

```sh
uv sync --frozen --project research/fsm_author --python 3.12
research/fsm_author/.venv/bin/python -m pytest -q research/fsm_author/tests
```

The [engineering report](../fsm-author-engineering-results.md) describes the
synthetic runtime and causal-interface checks. The [measured protocol](../fsm-author-bla-protocol.md)
specifies the restricted data split and single BLA28 fit. Never run the upstream
notebooks unchanged for this comparison: their data scope includes our reserved
records, and periodic warmup differs from the causal request interface.

Fit commands require the registered source/prerequisite hashes and exclusive
new output directories. A historical original run is not silently rerun by the
test command. See [third-party notices](THIRD_PARTY.md) and [license](LICENSE).
