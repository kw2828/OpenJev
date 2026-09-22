The fixed-data comparison completed all fifteen final models across five families and three paired seeds, with 52,800 optimizer updates. All models, predictions and raw records are retained.

The spatial model did not improve mean exposed-validation MSE: it was 2.246% above the matched neighbor-free model and 4.934% above the statistics baseline. Statistics had the lowest validation MSE in every paired seed. Dense128 fit training data best and had the highest validation MSE. No autonomous or novel-architecture advantage is established.

Independent saved-model replay checked 100,470 model-row pairs, matching saved NumPy predictions exactly. Original worker time was 1,513.80 seconds, with a separate 44.99-second audit. These are scalar-fitting and audit costs, not autonomous-controller latency.

The 61,800,541-byte archive contains 337 verified original files, including all fifteen initial/final checkpoints, prediction arrays, complete training journals, audit evidence, plots, 209 frozen scientific sources and applicable licenses. Read RESTORE.md and dependencies.json before restoring; historical data/runtime dependencies remain separate. The strict numerical auditor retains original absolute paths.

Archive SHA-256: `56ba6868ff0359ec6a27d8631085776602b63227c14fd63a0fddff0f72b51662`.

[Full results](https://github.com/kw2828/OpenJev/blob/main/research/otto-spatial-study-results.md) · [Protocol](https://github.com/kw2828/OpenJev/blob/main/research/otto-spatial-training-protocol.md)
