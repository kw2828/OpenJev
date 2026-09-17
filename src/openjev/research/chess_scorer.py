"""All-legal-move chess adapter for OpenJev's pinned, unchanged MLX model.

The public text protocol remains unchanged. Chess uses a separate, tokenizer-
verified label pool; probabilities are conditional move preferences, not odds
of winning. No engine or generated reasoning is used by this policy.
"""

import hashlib
import itertools
import json
import string
import time

import numpy as np

from openjev.decisions import MAX_TOKENS, MODEL_ID, MODEL_REVISION, MLXScorer
from openjev.research.chess_arena import chess_record

SYSTEM = (
    "You play chess. Choose the strongest legal move for the side to move. "
    "Use the supplied board and move history. All candidate moves are legal. "
    "Respond only with the selected candidate label. Do not explain."
)
PROTOCOL = {
    "version": "chess-candidate-label-v1",
    "system": SYSTEM,
    "ordering": "UCI ascending; fixed single-token uppercase labels",
    "labels": "A-Z then AA-ZZ, retaining unique single tokens in the pinned tokenizer",
    "readout": "one forward pass, next-token logits restricted to all legal candidates",
    "model": MODEL_ID,
    "revision": MODEL_REVISION,
    "max_input_tokens": MAX_TOKENS,
    "history_plies": 16,
    "search": "none",
    "calibration": "none",
}


def label_pool(tokenizer):
    strings = list(string.ascii_uppercase) + [
        "".join(x) for x in itertools.product(string.ascii_uppercase, repeat=2)
    ]
    labels, ids = [], []
    for label in strings:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in ids:
            labels.append(label)
            ids.append(encoded[0])
    if len(labels) < 218:
        raise ValueError("Pinned tokenizer cannot represent every possible legal move count")
    return labels, ids


def chess_messages(record, labels):
    candidates = sorted(record["candidates"], key=lambda c: c["id"])
    if not candidates or len(candidates) > len(labels):
        raise ValueError("No legal candidates or insufficient single-token labels")
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Duplicate move IDs")
    payload = {"context": record["context"], "question": record["question"],
               "candidates": {labels[i]: c["description"] for i, c in enumerate(candidates)}}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload)}], candidates


def move_distribution(logits, token_ids, candidates):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError("Invalid model logits")
    selected_ids = np.asarray(token_ids[:len(candidates)], dtype=int)
    if len(selected_ids) != len(candidates) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Nonunique or missing labels")
    selected = logits[selected_ids]
    probabilities = np.exp(selected - selected.max())
    probabilities /= probabilities.sum()
    normalizer = np.exp(logits - logits.max()).sum()
    mass = np.exp(logits[selected_ids] - logits.max()).sum() / normalizer
    return {"choice": candidates[int(probabilities.argmax())]["id"],
            "probabilities": {c["id"]: float(p) for c, p in zip(candidates, probabilities, strict=True)},
            "candidate_token_mass": float(mass),
            "candidate_count": len(candidates),
            "entropy_nats": float(-sum(p * np.log(p) for p in probabilities if p > 0))}


class ChessMLXPolicy:
    """Load once and call sequentially on the same owner thread."""

    name = "OpenJev / Qwen3-4B direct"

    def __init__(self, model_path=None):
        if model_path is not None:
            raise ValueError("This adapter only uses the pinned OpenJev cached checkpoint")
        start = time.perf_counter()
        self.scorer = MLXScorer()
        self.labels, self.label_ids = label_pool(self.scorer.tokenizer)
        self.load_ms = (time.perf_counter() - start) * 1000
        self.metadata = {**PROTOCOL, "backend": "mlx-4bit", "load_ms": self.load_ms,
                         "generated_tokens": 0,
                         "label_pool_sha256": hashlib.sha256(json.dumps(
                             list(zip(self.labels, self.label_ids, strict=True))).encode()).hexdigest()}

    def __call__(self, board):
        start = time.perf_counter()
        record = chess_record(board, history_plies=PROTOCOL["history_plies"])
        messages, candidates = chess_messages(record, self.labels)
        scorer = self.scorer
        prompt = scorer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = scorer.tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) > MAX_TOKENS:
            raise ValueError(f"Chess prompt exceeds {MAX_TOKENS} tokens; no move was dropped")
        mx = scorer.mx
        hidden = scorer.model.model(mx.array([tokens]))[:, -1:, :]
        logits = (scorer.model.model.embed_tokens.as_linear(hidden)
                  if scorer.model.args.tie_word_embeddings else scorer.model.lm_head(hidden))[0, -1]
        logits = logits.astype(mx.float32)
        mx.eval(logits)
        answer = move_distribution(np.array(logits), self.label_ids, candidates)
        return {**answer, "input_tokens": len(tokens), "model_forward_passes": 1,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "policy": self.name, "probability_semantics": "uncalibrated legal-move preferences"}

    def close(self):
        pass
