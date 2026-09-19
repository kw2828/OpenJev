"""Visualize the saved capacity decision, without rerunning an encoder."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
CAPACITY = ROOT / "output/dialogue-joint-v1/capacity-01/completed.json"
PIN = "7f8174f3afb7c81e29740e373f83224d0ca723a14d847f667e12c59aa1d779b3"


def main():
    raw = CAPACITY.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PIN
    record = json.loads(raw)
    projection = record["projection"]
    assert record["status"] == "completed" and projection["encoding_permitted"] is False
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Candidate-conditioned encoding: cost screen not passed", fontsize=17, y=.98)
    fig.text(.5, .905, "512 texts encoded; full encoding and model training not launched", ha="center", fontsize=11)
    left, right = axes
    left.bar([0, 1], [record["wall_seconds"] / 60, projection["projected_total_seconds"] / 60],
             color=["#8195a5", "#bf5b48"], width=.58)
    left.axhline(12, color="#353535", linestyle="--", label="12 min admission limit")
    left.axhline(15, color="#888888", linestyle=":", label="15 min execution cap")
    left.set_xticks([0, 1], ["Measured capacity probe\n(includes full token counting)", "Projected full encoding\n(not measured)"])
    left.set_ylabel("Minutes")
    left.set_ylim(0, 26)
    left.text(0, record["wall_seconds"] / 60 + .6, f"{record['wall_seconds']:.2f} s", ha="center")
    left.text(1, projection["projected_total_seconds"] / 60 + .6,
              f"{projection['projected_total_seconds'] / 60:.2f} min", ha="center", fontweight="bold")
    left.legend(loc="upper left", fontsize=9, frameon=False)
    size = projection["projected_cache_bytes"] / 1024**3
    right.bar([0], [size], width=.5, color="#397e8a")
    right.axhline(4, color="#353535", linestyle="--", label="4 GiB cache limit")
    right.set_xticks([0], ["Projected full cache\n(size limit passed)"])
    right.set_ylabel("GiB")
    right.set_xlim(-.8, .8)
    right.set_ylim(0, 5)
    right.text(0, size + .12, f"{size:.2f} GiB", ha="center", fontweight="bold")
    right.legend(loc="upper left", fontsize=9, frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#ececec")
        axis.set_axisbelow(True)
    fig.text(.5, .09, "Time projection uses 4 synchronized MPS forwards over the fixed first 512 unique texts.",
             ha="center", fontsize=10)
    fig.text(.5, .052, "This cost projection is not a measured full-run latency or evidence about model accuracy.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.07, right=.97, top=.82, bottom=.25, wspace=.3)
    out = Path(__file__).parent
    image = out / "capacity.png"
    if image.exists() or (out / "receipt.json").exists():
        raise FileExistsError("Figure destination already exists")
    fig.savefig(image, dpi=160)
    plt.close(fig)
    receipt = {"status": "completed", "capacity_completed_sha256": PIN,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
               "encoder_calls": 0, "scope": "Saved capacity measurement and projection only"}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
