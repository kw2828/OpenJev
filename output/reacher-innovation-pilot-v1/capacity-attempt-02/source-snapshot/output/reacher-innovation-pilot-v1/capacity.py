"""Single full-shape synthetic qualification; no real corpus or scientific RNG."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
import traceback
from pathlib import Path

import torch

from openjev.research.reacher_innovation_context import VARIANTS, InnovationContextWorldModel
from openjev.research.reacher_innovation_pilot import PilotConfig, evaluate_development, fit_one
from openjev.research.reacher_objective_training import canonical_tensor_hash

OUT = Path("output/reacher-innovation-pilot-v1/capacity-attempt-02")


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def data(gap):
    angle = torch.linspace(-0.4, 0.8, 32 * 51 * 2).reshape(32, 51, 2)
    packets = torch.zeros(32, 51, 8)
    packets[..., :2], packets[..., 2:4] = angle.cos(), angle.sin()
    packets[..., 4:6] = torch.tensor([0.1, -0.09])
    packets[..., 6] = 1
    for start in (8, 28):
        packets[:, start:start + gap, :4] = 0
        packets[:, start:start + gap, 6] = 0
        packets[:, start:start + gap, 7] = torch.arange(1, gap + 1).float() * 0.02
    return {"packets": packets,
            "commands": torch.linspace(-0.6, 0.6, 32 * 50 * 2).reshape(32, 50, 2),
            "rewards": torch.linspace(-0.8, -0.1, 32 * 50).reshape(32, 50)}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    begin, deadline = time.monotonic(), time.monotonic() + 120
    runtime = {"torch": str(torch.__version__), "platform": platform.platform(), "dtype": "float32", "torch_threads": 1}
    sources = json.loads(Path("evidence/reacher-two-observation-study-v1/protocol/freeze.json").read_text())["source_sha256"]
    for path, digest in sources.items():
        assert sha(path) == digest, path
    sources.update({str(path): sha(path) for path in [Path(__file__).resolve().relative_to(Path.cwd().resolve()),
        Path("src/openjev/research/reacher_innovation_context.py"),
        Path("src/openjev/research/reacher_innovation_loss.py"),
        Path("src/openjev/research/reacher_innovation_pilot.py")]})
    write(OUT / "started.json", {"engineering_only": True, "seed": 410, "runtime": runtime,
                                "source_sha256": sources, "cap_seconds": 120})
    (OUT / "as-run-driver.py").write_bytes(Path(__file__).read_bytes())
    for name in sources:
        target = OUT / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    records = []
    try:
        ordinary, shifted, orders = data(6), data(10), torch.arange(96).reshape(1, 96)
        training = {key: value.repeat(3, 1, 1) if value.ndim == 3 else value.repeat(3, 1)
                    for key, value in ordinary.items()}
        for variant in VARIANTS:
            config = PilotConfig(variant, 64, 16, 0.02, 0.05, 0.1, 1e-4, 4., 1, 32, 0.1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(410)
                model = InnovationContextWorldModel(**config.model_kwargs()).float()
                weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            fit_begin = time.monotonic()
            fit = fit_one(weights, orders, training, OUT / variant / "fit", config=config,
                source_sha256=sources, runtime=runtime, expected_initial_sha256=canonical_tensor_hash(weights),
                expected_orders_sha256=canonical_tensor_hash({"orders": orders}),
                expected_data_sha256=canonical_tensor_hash(training), deadline=deadline)
            fit_seconds = time.monotonic() - fit_begin
            fitted = torch.load(OUT / variant / "fit" / "weights.pt", weights_only=True)
            evaluations = []
            for panel, values in (("six", ordinary), ("ten", shifted)):
                tick = time.monotonic()
                result = evaluate_development(fitted, values, OUT / variant / panel, config=config,
                    source_sha256=sources, runtime=runtime, expected_weights_sha256=fit["final_weights_sha256"],
                    expected_data_sha256=canonical_tensor_hash(values), deadline=deadline)
                assert all(row["post_reacquisition_complete_two_roots"] for row in result["per_episode"])
                evaluations.append({"panel": panel, "episodes": 32, "wall_seconds": time.monotonic() - tick,
                                    "complete_two_root_coverage": True})
            records.append({"variant": variant, "fit_wall_seconds": fit["wall_seconds"],
                            "whole_fit_call_wall_seconds": fit_seconds, "training_episodes": 96,
                            "optimizer_steps": fit["counts"]["optimizer_steps"], "evaluations": evaluations})
        for path, digest in sources.items():
            assert sha(path) == digest, path
        assert time.monotonic() < deadline
        members = {str(p.relative_to(OUT)): {"sha256": sha(p), "bytes": p.stat().st_size}
                   for p in OUT.rglob("*") if p.is_file()}
        write(OUT / "completed.json", {"engineering_only": True, "records": records,
            "wall_seconds": time.monotonic() - begin, "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_units": "bytes on this macOS host", "members": members,
            "limits": "Twelve synthetic updates and eight synthetic evaluation batches only; no efficacy or speed comparison."})
        assert time.monotonic() < deadline
        print(json.dumps({"status": "engineering_completed", "seconds": time.monotonic() - begin}))
    except BaseException as error:
        original_traceback = traceback.format_exc()
        try:
            if (OUT / "completed.json").exists():
                (OUT / "completed.json").rename(OUT / "invalid-completion.json")
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note("Terminal demotion failed: " + repr(preservation_error))
        try:
            write(OUT / "failed.json", {"error": repr(error), "traceback": original_traceback,
                                        "seconds": time.monotonic() - begin, "automatic_retry": False})
        except BaseException as preservation_error:  # noqa: BLE001
            error.add_note("Failure preservation failed: " + repr(preservation_error))
        raise


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
