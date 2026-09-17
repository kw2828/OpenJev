"""Publish every mate-arena game unchanged after the frozen report audit."""

import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = ROOT / "evidence/chess-mate-arena-v1/plan.json"
    execution = ROOT / "runs/chess-mate-arena-v1/execution"
    report = ROOT / "runs/chess-mate-arena-v1/report"
    out = ROOT / "evidence/chess-mate-arena-v1/results"
    if out.exists():
        raise FileExistsError(out)
    spec = importlib.util.spec_from_file_location("mate_arena", ROOT / "scripts/chess_mate_arena.py")
    arena = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arena)
    with tempfile.TemporaryDirectory() as temporary:
        verified = arena.report(plan, execution, Path(temporary) / "report")
    if verified != json.loads((report / "summary.json").read_text()):
        raise ValueError("Arena summary does not reproduce")
    inputs = {str(p.relative_to(execution)): p for p in sorted(execution.iterdir()) if p.is_file()}
    before = {name: sha(path) for name, path in inputs.items()}
    out.mkdir()
    for name, path in inputs.items():
        shutil.copyfile(path, out / name)
        if sha(out / name) != before[name] or sha(path) != before[name]:
            raise ValueError("Arena copy differs from original")
    shutil.copytree(report, out / "report")
    files = {
        str(p.relative_to(out)): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    receipt = {
        "status": "completed",
        "plan_sha256": sha(plan),
        "publisher_sha256": sha(__file__),
        "arena_runner_sha256": sha(ROOT / "scripts/chess_mate_arena.py"),
        "report_reproduced": True,
        "new_games_played": False,
        "files": files,
    }
    with (out / "publication.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "published", "games": verified["games"], "files": len(files)}))


if __name__ == "__main__":
    main()
