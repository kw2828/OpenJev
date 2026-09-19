"""Render the first fixed comparison from saved public observations only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(attempt: Path, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    modes = ("full", "last32")
    labels = ("Full public memory", "Last 32 transitions")
    payloads, hashes = [], {}
    for mode in modes:
        source = attempt / "episodes" / f"000-{mode}-order0.npz"
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        with np.load(source, allow_pickle=False) as data:
            payloads.append({name: data[name] for name in ("frames", "observations", "rewards")})
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    title = ImageFont.load_default(size=24)
    frames = []
    maximum = max(len(data["frames"]) for data in payloads)
    selected = sorted(set(range(0, maximum, 2)) | {maximum - 1})
    for step in selected:
        image = Image.new("RGB", (860, 438), "#0e1726")
        draw = ImageDraw.Draw(image)
        draw.text((24, 16), "Mystery Path: what does the controller remember?", font=title, fill="white")
        draw.text((24, 49), "First fixed layout, order 0. Public discoveries only. Rule-based controllers.", font=small, fill="#aabbcc")
        for column, (data, label) in enumerate(zip(payloads, labels)):
            index = min(step, len(data["frames"]) - 1)
            left = 24 + column * 424
            draw.text((left, 83), label, font=font, fill="#e7efff")
            known = {}
            start = 0 if column == 0 else max(0, index - 32)
            for obs in data["observations"][start:index+1]:
                x, y, _, failure = (int(v) for v in obs)
                known[x, y] = bool(failure)
            for x in range(7):
                for y in range(7):
                    color = "#233044" if (x, y) not in known else ("#c45d68" if known[x, y] else "#438b7c")
                    draw.rectangle((left+x*36, 119+y*36, left+x*36+33, 119+y*36+33), fill=color)
            x, y, heading, failed = (int(v) for v in data["observations"][index])
            cx, cy = left+x*36+16, 119+y*36+16
            draw.ellipse((cx-10, cy-10, cx+10, cy+10), fill="#ffd36c")
            dx, dy = ((0,-1),(-1,0),(0,1),(1,0))[heading]
            draw.line((cx, cy, cx+dx*14, cy+dy*14), fill="white", width=3)
            raw = Image.fromarray(data["frames"][index].transpose(1,0,2))
            image.paste(raw, (left+280, 151))
            draw.text((left+270, 122), "Actual view", font=small, fill="#aabbcc")
            falls = int(data["observations"][1:index+1, 3].sum())
            ended = index == len(data["frames"])-1
            status = ("Goal reached" if data["rewards"].sum() == 1 else "Action limit") if ended else ("Fell off route" if failed else "Exploring")
            draw.text((left, 383), f"Action {index}/128 | Falls {falls} | {status}", font=small, fill="white")
        draw.text((24, 412), "Green: observed safe. Red: observed unsafe. Gray: unknown or forgotten. Replay slowed for viewing.", font=small, fill="#aabbcc")
        frames.append(image)
    frames[0].save(out / "episode.gif", save_all=True, append_images=frames[1:], duration=220, loop=0)
    frames[len(frames)//2].save(out / "episode.png")
    receipt = {"selection": "layout_index=0, order_index=0, full versus last32; fixed before reading outcomes",
               "sources": hashes, "rendered_frames": len(frames), "new_environment_calls": 0,
               "note": "Maps reconstructed from public frames only; no hidden path or seed input."}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.attempt, args.out)
