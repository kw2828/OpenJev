"""ChessLFM's published board-token protocol, without its demo's search.

Independent adapter from the model card and inspected Space encoding. Upstream
weights retain the LFM license. This is a convolution/attention hybrid, not a
new OpenJev recurrent world model. No remote code is executed.
"""

import json
import time

import numpy as np

MODEL_ID = "mlabonne/LFM2.5-230M-Chess"
MODEL_REVISION = "341bfd2ea4b696cda7d2016d4a721f27eb9cf34d"


def board_tokens(board):
    if not board.is_valid():
        raise ValueError("Invalid chess position")
    fields = board.fen().split()
    cells = []
    for char in fields[0].replace("/", ""):
        cells.extend(["."] * int(char) if char.isdigit() else [char])
    # Repetition token counts prior occurrences, capped at two.
    repeats = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0
    history = [m.uci() for m in board.move_stack[-8:]]
    tokens = ["<|pos|>"] + [f"<c:{c}>" for c in cells]
    tokens += [f"<stm:{fields[1]}>", f"<cast:{fields[2]}>", f"<ep:{fields[3][0]}>",
               f"<hm:{min(board.halfmove_clock, 100) // 4}>", f"<rep:{repeats}>", "<|hist|>"]
    tokens += ["<m:0000>"] * (8-len(history)) + [f"<m:{m}>" for m in history] + ["<|eval|>"]
    if len(tokens) != 80:
        raise ValueError("ChessLFM requires exactly 80 board tokens")
    return tokens


class ChessLFMPolicy:
    name = "ChessLFM 230M / direct"

    def __init__(self, model_path=None, device="cpu", threads=2):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM

        start = time.perf_counter()
        path = model_path or snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.jinja"],
        )
        from pathlib import Path
        self.token_ids = json.loads((Path(path)/"token_ids.json").read_text())
        torch.set_num_threads(threads)
        self.torch, self.device = torch, device
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
            attn_implementation="eager",
        ).to(device).eval()
        self.value_ids = [self.token_ids[f"<v:{i}>"] for i in range(64)]
        self.load_ms = (time.perf_counter()-start)*1000
        self.metadata = {"model": MODEL_ID, "revision": MODEL_REVISION,
                         "backend": f"torch-{device}-float32", "threads": threads,
                         "search": "none", "forward_passes_per_move": 2,
                         "protocol": "published-80-token-value-then-move",
                         "load_ms": self.load_ms, "calibration": "none"}

    def __call__(self, board):
        start = time.perf_counter()
        moves = sorted(m.uci() for m in board.legal_moves)
        if not moves:
            raise ValueError("No legal moves")
        ids = [self.token_ids[t] for t in board_tokens(board)]
        torch = self.torch
        with torch.inference_mode():
            value_logits = self.model(input_ids=torch.tensor([ids], device=self.device),
                                      use_cache=False, logits_to_keep=1).logits[0, -1]
            value_bin = int(value_logits[self.value_ids].argmax())
            ids += [self.value_ids[value_bin], self.token_ids["<|bestmove|>"]]
            logits = self.model(input_ids=torch.tensor([ids], device=self.device),
                                use_cache=False, logits_to_keep=1).logits[0, -1]
            legal_logits = logits[[self.token_ids[f"<m:{m}>"] for m in moves]]
            probabilities = legal_logits.softmax(0).float().cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite ChessLFM probabilities")
        return {"choice": moves[int(probabilities.argmax())],
                "probabilities": dict(zip(moves, map(float, probabilities), strict=True)),
                "candidate_count": len(moves), "value_bin": value_bin,
                "predicted_win_probability": (value_bin+.5)/64,
                "latency_ms": (time.perf_counter()-start)*1000,
                "model_forward_passes": 2, "policy": self.name,
                "probability_semantics": "uncalibrated legal-move preferences"}

    def close(self):
        pass
