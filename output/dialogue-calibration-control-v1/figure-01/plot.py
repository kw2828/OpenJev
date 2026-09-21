"""All replay timings and the unchanged failed cost gate; no model calls."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/dialogue-calibration-control-v1/qualification-01/completed.json"
AUDIT = ROOT / "output/dialogue-calibration-control-v1/qualification-review-01/receipt.json"
PINS = {RUN: "ca06666f1849a49c3e230f8a1fa6f485f5214dbdbe36c9f7fccde1a2cf2e2a94",
        AUDIT: "f8505e88a63c61b803f7dda51da8ebd9839fb583b198ea3ee076501c09afae9c"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    for path, pin in PINS.items():
        assert sha(path) == pin
    result = json.loads(RUN.read_text())
    audit = json.loads(AUDIT.read_text())
    assert audit["status"] == "completed" and audit["agreement"] is True
    assert result["parity_passed"] is True and result["completed_forwards"] == 24
    projection = result["projection"]
    assert projection["admitted"] is False
    labels, first, largest = [], [], []
    for fit in result["fits"]:
        labels.append(fit["fit_id"])
        assert len(fit["cases"]) == 2
        first.append(fit["cases"][0]["case_seconds"] * 1000)
        largest.append(fit["cases"][1]["case_seconds"] * 1000)
    overhead = projection["loading_seconds"] + projection["preparation_seconds"]
    keys = ("encoder_calls", "padded_attention_positions", "real_question_updates")
    totals = [(projection["measures"][key]["projected_seconds"] + overhead) / 60 for key in keys]
    assert math.isclose(max(totals) * 60, projection["total_seconds"], abs_tol=1e-10)
    values = {"fit_order": labels, "first_case_ms": first, "maximum_attention_case_ms": largest,
              "projection_measures": list(keys), "projection_minutes_including_preparation_and_loads": totals,
              "threshold_minutes": projection["threshold_seconds"] / 60,
              "raw_endpoint_maximum_log_difference": 0.0, "raw_endpoint_maximum_probability_difference": 0.0,
              "scope": "All 24 measured cases. Projected costs are estimates, not measured full inference. "
                       "First-case outlier is descriptive; no startup cause established."}
    write(OUT / "source-values.json", values)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(12)
    left.scatter(first, y - .12, marker="o", s=40, color="#2563eb", label="First DEV dialogue: 9 recurrent updates")
    left.scatter(largest, y + .12, marker="s", s=35, color="#0f766e", label="Maximum-attention dialogue: 42 updates")
    left.set_yticks(y, labels=[s.replace("frozen_", "F / ").replace("trainable_", "T / ").replace("-", " / ") for s in labels])
    left.invert_yaxis()
    left.set_xlim(0, 180)
    left.set_xlabel("Measured full case duration (milliseconds)")
    left.set_title("All twelve checkpoints, both fixed dialogues", loc="left", fontweight="bold", pad=14)
    left.grid(axis="x", alpha=.18)
    left.set_axisbelow(True)
    left.legend(loc="lower right", fontsize=8, frameon=False)
    left.annotate("First replay: 162.75 ms", (first[0], -.12), xytext=(95, 1.2), fontsize=9,
                  arrowprops={"arrowstyle": "->", "color": "#64748b"})
    names = ("Encoder calls", "Attention positions", "Recurrent updates")
    right.barh(np.arange(3), totals, height=.55, color=["#94a3b8", "#94a3b8", "#dc2626"])
    right.set_yticks(np.arange(3), labels=names)
    right.invert_yaxis()
    right.axvline(30, color="#111827", linestyle="--", linewidth=1.3)
    right.set_xlim(0, 103)
    right.set_xlabel("Projected full-calibration duration (minutes)")
    right.set_title("Maximum projection rejects admission", loc="left", fontweight="bold", pad=14)
    for i, value in enumerate(totals):
        right.text(value + 1, i, f"{value:.2f}", va="center", fontweight="bold")
    right.text(31.5, -.48, "30-minute limit", fontsize=9)
    right.grid(axis="x", alpha=.18)
    right.set_axisbelow(True)
    fig.suptitle("OpenJev calibration control: exact replay, failed cost gate", x=.02, ha="left", fontsize=16, fontweight="bold")
    fig.text(.02, .02, "24 / 24 replays agree exactly across 552 endpoints. Full calibration was not launched; no temperatures were fitted.\n"
             "F = frozen encoder; T = trained encoder. Every projection includes all 12 loads and preparation. No architecture or calibration gain claimed.",
             fontsize=10, color="#334155")
    fig.tight_layout(rect=(0, .12, 1, .94), w_pad=3)
    png = OUT / "replay-cost-gate.png"
    assert not png.exists()
    fig.savefig(png, dpi=180, facecolor="white")
    plt.close(fig)
    write(OUT / "receipt.json", {"status": "completed", "source_sha256": sha(Path(__file__)),
          "inputs": {str(p.relative_to(ROOT)): pin for p, pin in PINS.items()},
          "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in (png, OUT / "source-values.json")},
          "model_calls": 0, "scope": values["scope"]})
    print(json.dumps({"status": "completed", "png": str(png), "sha256": sha(png)}))


if __name__ == "__main__":
    main()
