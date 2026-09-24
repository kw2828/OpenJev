# Same GRU, conditional cost labels: audited DEV_FAIL

Six fresh GRU fits compare sampled labels with the mean of the same 32-label
bank, under paired initialization, case order and optimizer work. Averaging
lowers mean teacher-cost decision regret 4.89% in the training regime but raises
it 6.15% in the unseen sensing regime. Both settings fail the predeclared
five-percent margin and all-seed consistency requirements.

The study retains 329 TRAIN and 173 DEV prefixes from 768 fresh starts.
Collection, fitting and the independent saved-output audit closed successfully.
Shared acquisition used 30,512 teacher endpoint annotations; the three workers
took 1,285.933 seconds in total. No autonomous gameplay, calibration, RL or
architectural advantage is established.

The evidence archive contains all six new checkpoints, the new label banks,
original phase logs and receipts, registered sources and inputs, independent
audit, qualification failures, report and README media. The manifest records
each member's provenance and hash. Absolute paths and external runtime
dependencies are documented; this archive is evidence, not a portable installer
or authorization to rerun the frozen experiment.

- [Complete results](https://github.com/kw2828/OpenJev/blob/main/research/otto-conditional-label-results.md)
- [Protocol](https://github.com/kw2828/OpenJev/blob/main/research/otto-conditional-label-protocol.md)
- [Separate prospective recurrent component](https://github.com/kw2828/OpenJev/blob/main/research/otto-observation-operator-status.md)

Original closure SHA256:
`0af9ba9d296b9558b1a969e28e3925b739cee4a64aa2bb9ed1f2b1479c7a9e53`.
