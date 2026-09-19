"""Matched, request-local Qwen scoring variants for a systems experiment.

Right padding is safe here because we read only each last real token under
causal attention. Every suffix cache is discarded, never continued past padding.
No production API changes, output generation, prompt changes or weight updates.
"""

import time
from dataclasses import dataclass

import numpy as np

from openjev.decisions import MAX_TOKENS, messages_for, summarize_logits

METHODS = ("serial", "batch", "shared_serial", "shared_batch")


def common_prefix_length(sequences):
    """Leave at least one real suffix token, including for a single question."""
    if not sequences or any(not seq for seq in sequences):
        raise ValueError("Nonempty token sequences are required")
    limit = min(map(len, sequences)) - 1
    for i in range(limit):
        if any(seq[i] != sequences[0][i] for seq in sequences[1:]):
            return i
    return limit


def right_pad(sequences, pad_id=0):
    if not sequences or any(not seq for seq in sequences):
        raise ValueError("Nonempty token sequences are required")
    lengths = np.array([len(seq) for seq in sequences], dtype=np.int32)
    tokens = np.full((len(sequences), int(lengths.max())), pad_id, dtype=np.int32)
    for i, seq in enumerate(sequences):
        tokens[i, :len(seq)] = seq
    return tokens, lengths


@dataclass
class Work:
    calls: int = 0
    prefill_calls: int = 0
    branch_calls: int = 0
    input_token_slots: int = 0
    prefix_tokens: int = 0
    padding_token_slots: int = 0


class SharedPrefixExperiment:
    def __init__(self, scorer):
        self.scorer = scorer
        self.mx = scorer.mx
        self.model = scorer.model

    def prepare(self, request):
        prepared = []
        for question in request.questions:
            messages, ordered = messages_for(request.context, question)
            prompt = self.scorer.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            tokens = self.scorer.tokenizer.encode(prompt, add_special_tokens=False)
            if not tokens or len(tokens) > MAX_TOKENS:
                raise ValueError("Invalid input length; nothing was truncated")
            prepared.append((question, ordered, tokens))
        return prepared

    def logits(self, sequences, method):
        """Return full vocabulary logits, and count actual model invocations."""
        from mlx_lm.models.cache import KVCache, make_prompt_cache

        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        # Validate even the no-cache path before touching the model.
        prefix_length = common_prefix_length(sequences)
        mx = self.mx
        work = Work()

        def forward(rows, cache=None):
            padded, lengths = right_pad(rows)
            work.calls += 1
            work.branch_calls += 1
            work.input_token_slots += int(padded.size)
            work.padding_token_slots += int(padded.size - lengths.sum())
            h = self.model.model(mx.array(padded), cache=cache)
            last = h[mx.arange(len(rows)), mx.array(lengths - 1), :][:, None, :]
            out = (self.model.model.embed_tokens.as_linear(last)
                   if self.model.args.tie_word_embeddings else self.model.lm_head(last))[:, 0]
            out = out.astype(mx.float32)
            mx.eval(out)
            return np.array(out)

        if method == "serial":
            return np.concatenate([forward([seq]) for seq in sequences]), work
        if method == "batch":
            return forward(sequences), work
        if prefix_length == 0:
            if method == "shared_serial":
                return np.concatenate([forward([seq]) for seq in sequences]), work
            return forward(sequences), work

        prefix = make_prompt_cache(self.model)
        if any(type(cache) is not KVCache for cache in prefix):
            raise TypeError("This experiment supports ordinary Qwen KVCache only")
        self.model.model(mx.array([sequences[0][:prefix_length]]), cache=prefix)
        # All cached layers must finish before the suffix. Discard unused final
        # prefix norm, matching the fact that no vocabulary readout is requested.
        mx.eval([cache.state for cache in prefix])
        work.calls += 1
        work.prefill_calls += 1
        work.prefix_tokens = prefix_length
        work.input_token_slots += prefix_length
        saved = [cache.state for cache in prefix]

        def branch(batch_size):
            result = []
            for keys, values in saved:
                # state is trimmed to the real prefix. A nonempty suffix forces
                # KVCache to allocate a new buffer before writing, so the saved
                # prefix cannot be contaminated by another question's suffix.
                cache = KVCache()
                cache.state = (mx.repeat(keys, batch_size, axis=0),
                               mx.repeat(values, batch_size, axis=0))
                assert cache.offset == prefix_length
                result.append(cache)
            return result

        suffixes = [seq[prefix_length:] for seq in sequences]
        if method == "shared_serial":
            out = np.concatenate([forward([seq], branch(1)) for seq in suffixes])
        else:
            out = forward(suffixes, branch(len(suffixes)))
        return out, work

    def score(self, request, method):
        """Wall time includes tokenization, cache branching and CPU readback."""
        mx = self.mx
        mx.synchronize()
        mx.reset_peak_memory()
        baseline_memory = mx.get_active_memory()
        started = time.perf_counter()
        prepared = self.prepare(request)
        logits, work = self.logits([row[2] for row in prepared], method)
        answers = [summarize_logits(values, self.scorer.label_ids, ordered,
                                   question.id, len(tokens), 0.)
                   for values, (question, ordered, tokens) in zip(logits, prepared)]
        # JSON-ready formatting and readback are charged, without attributing
        # overlapping request work to individual questions.
        answer_rows = [answer.model_dump(exclude={"latency_ms"}) for answer in answers]
        label_logits = [values[np.array(self.scorer.label_ids[:len(ordered)])].tolist()
                        for values, (_, ordered, _) in zip(logits, prepared)]
        mx.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        return {"answers": answer_rows, "candidate_logits": label_logits, "latency_ms": elapsed,
                "peak_active_bytes": mx.get_peak_memory(),
                "baseline_active_bytes": baseline_memory,
                "cached_allocator_bytes": mx.get_cache_memory(),
                "work": vars(work), "questions_sequential": method in ("serial", "shared_serial")}
