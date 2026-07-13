"""Stage 1: Chronological consistency validation."""

import logging

import torch
import tiktoken

logger = logging.getLogger(__name__)

from .model_loader import encode as _encode, pad_token as _pad_token

_default_tokenizer = tiktoken.get_encoding("gpt2")


def _score_prompt(model, device, prompt, phrase, tokenizer=None):
    tokenizer = tokenizer or _default_tokenizer
    prompt_tokens = _encode(tokenizer, prompt)
    phrase_tokens = _encode(tokenizer, " " + phrase)

    if not prompt_tokens or not phrase_tokens:
        return -10.0

    total = 0.0
    current_tokens = list(prompt_tokens)

    with torch.no_grad():
        for token_id in phrase_tokens:
            input_ids = torch.tensor([current_tokens]).to(device)
            logits = model(input_ids)
            probs = torch.nn.functional.softmax(logits[0, -1, :], dim=-1)
            total += torch.log(probs[token_id] + 1e-10).item()
            current_tokens.append(token_id)

    return total / len(phrase_tokens)


def _score_batch(model, device, items, tokenizer=None):
    """Score all items in batched forward passes."""
    if not items:
        return []

    tokenizer = tokenizer or _default_tokenizer
    pad_token = _pad_token(tokenizer)

    all_prompt_tokens = []
    all_phrase_tokens = []
    for item in items:
        prompt_tokens = _encode(tokenizer, item["prompt"])
        phrase_tokens = _encode(tokenizer, " " + item["phrase"])
        all_prompt_tokens.append(prompt_tokens if prompt_tokens else [pad_token])
        all_phrase_tokens.append(phrase_tokens if phrase_tokens else [])

    max_phrase_len = max(len(phrase_tokens) for phrase_tokens in all_phrase_tokens) if all_phrase_tokens else 0
    current = [list(prompt_tokens) for prompt_tokens in all_prompt_tokens]
    totals = [0.0] * len(items)
    counts = [len(phrase_tokens) for phrase_tokens in all_phrase_tokens]

    with torch.no_grad():
        for token_pos in range(max_phrase_len):
            active = [i for i in range(len(items)) if token_pos < len(all_phrase_tokens[i])]
            if not active:
                break

            seqs = [current[i] for i in active]
            max_len = max(len(s) for s in seqs)
            padded = [s + [pad_token] * (max_len - len(s)) for s in seqs]
            lengths = [len(s) for s in seqs]

            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            # No logit normalization here: logits/logits.std() amplifies flat
            # distributions (e.g. label-smoothed models), giving them an unfair
            # advantage. The tanh clamp in ChronoGPT (15*tanh(logits/15)) already
            # limits lm_head scaling exploits, and the cosine gate blocks copies.
            logits = model(input_ids)

            for batch_idx, item_idx in enumerate(active):
                pos = lengths[batch_idx] - 1
                probs = torch.nn.functional.softmax(logits[batch_idx, pos, :], dim=-1)
                expected_token = all_phrase_tokens[item_idx][token_pos]
                totals[item_idx] += torch.log(probs[expected_token] + 1e-10).item()
                current[item_idx].append(expected_token)

    return [t / c if c > 0 else -10.0 for t, c in zip(totals, counts)]


def evaluate(model, device, benchmark, tokenizer=None):
    """Validate chronological consistency. Returns (exceeded_threshold, median)."""
    items = benchmark.get("items", [])

    if not items:
        return False, -20.0

    threshold = benchmark.get("threshold", 0.10)
    epsilon = benchmark.get("epsilon", -6.0)

    scores = _score_batch(model, device, items, tokenizer)
    median = sorted(scores)[len(scores) // 2]
    failed = sum(1 for s in scores if s > epsilon)
    ratio = failed / len(scores)

    logger.debug(f"    median={median:.4f} failed={failed}/{len(scores)} ({ratio:.1%}) threshold={threshold:.0%}")

    return ratio > threshold, median
