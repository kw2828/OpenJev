"""Saved-cost heuristic for a prospective allocation; no model or task scores."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output/dialogue-observation-learning-v1"
PINS = {
    "cost-pilot-01/completed.json": "2ba77fed00551a111bdf427d9a53bac5c3edc96dcaa61acae6335c303007bc11",
    "cost-audit-01/result-01/receipt.json": "1b530eca81e650c9bf5874ce5ffb2f3aa9617a24733b4ef58f9dfe400acd611e",
    "preparation-02/plan.json": "4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


for name, pin in PINS.items():
    assert sha(BASE / name) == pin, name
done = read(BASE / "cost-pilot-01/completed.json")
summary_path = BASE / "cost-pilot-01/summary.json"
assert sha(summary_path) == done["files"]["summary.json"]["sha256"]
summary = read(summary_path)
audit = read(BASE / "cost-audit-01/result-01/receipt.json")
assert audit["agreement"] and audit["execution_completed_sha256"] == PINS["cost-pilot-01/completed.json"]
prepared = read(BASE / "preparation-02/plan.json")
epochs, effective = prepared["config"]["epochs"], prepared["config"]["effective_batch"]
train, dev = prepared["cohort_sizes"]["train"], prepared["cohort_sizes"]["dev"]
full, tail = epochs * (train // effective), epochs * int(train % effective != 0)
assert train % effective == 1 and (full, tail, dev) == (1260, 20, 2363)
arms = {}
for arm in prepared["config"]["arms"]:
    cells = [c for c in summary["cells"] if c["arm"] == arm]
    batches = [e["wall_seconds"] for c in cells if c["kind"] == "batch" for e in c["events"] if not e["warm"]]
    singles = [e["wall_seconds"] for c in cells if c["kind"] in ("single", "tail") for e in c["events"]]
    evaluation = [e["wall_seconds"] for c in cells if c["kind"] == "eval" for e in c["events"] if not e["warm"]]
    assert (len(batches), len(singles), len(evaluation)) == (6, 4, 3)
    arms[arm] = {
        "maximum_measured_full_batch_seconds": max(batches),
        "maximum_measured_single_update_seconds": max(singles),
        "maximum_measured_dev_dialogue_seconds": max(evaluation),
        "full_batches_per_fit": full, "single_tail_updates_per_fit": tail,
        "dev_dialogues_per_fit": dev,
        "estimated_seconds_per_fit_before_overhead": full * max(batches) + tail * max(singles) + dev * max(evaluation),
    }
raw = 3 * math.fsum(a["estimated_seconds_per_fit_before_overhead"] for a in arms.values())
headroom, overhead, allocation = .25, 600, 28800
planned = raw * (1 + headroom) + overhead
assert planned < allocation
checkpoint_bytes = 3 * sum(c["bytes"] for c in summary["checkpoints"])
result = {
    "status": "prospective_heuristic_only", "input_sha256": PINS, "script_sha256": sha(Path(__file__)),
    "arms": arms, "paired_seeds": 3, "estimated_base_seconds": raw,
    "multiplicative_headroom": headroom, "extra_fixed_overhead_seconds": overhead,
    "heuristic_with_headroom_seconds": planned, "prospective_wall_cap_seconds": allocation,
    "additional_cap_margin_seconds": allocation - planned,
    "checkpoint_bytes_from_measured_shapes": checkpoint_bytes,
    "prospective_rss_cap_bytes": 8 * 1024**3, "prospective_sampled_mps_driver_cap_bytes": 8 * 1024**3,
    "prospective_output_cap_bytes": 2 * 1024**3,
    "limits": [
        "Largest observed metadata-selected cases are not universal worst cases or a duration guarantee.",
        "Synthetic targets can produce different costs from task training; no accuracy is measured here.",
        "The envelope uses maxima across measured events, not the faster median or selected arm.",
        "Fixed overhead covers model loading, journals, evaluation artifacts, checksums and closure; extra multiplicative and cap headroom remain heuristic.",
        "Scientific admission still requires complete runner/reporter review, bound allocation and shared compute coordination.",
        "Preparation and qualification are separately paid work and are not included in the future execution cap.",
    ],
}
with (BASE / "cost-estimate-01.json").open("x") as stream:
    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
print(json.dumps({k: result[k] for k in ("estimated_base_seconds", "heuristic_with_headroom_seconds",
      "prospective_wall_cap_seconds", "checkpoint_bytes_from_measured_shapes")}))
