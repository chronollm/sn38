"""Stage 1: Chronological consistency validation."""

import logging

import torch

logger = logging.getLogger(__name__)


def _score_prompt(model, device, prompt, phrase):
    """Score a single prompt/phrase pair. Returns sum of log-probs."""
    return _score_batch(model, device, [{"prompt": prompt, "phrase": phrase}])[0]


def _score_batch(model, device, items):
    """Score all items in batched forward passes. Returns list of sum-of-log-probs."""
    if not items:
        return []

    pad_token = model.pad_token_id
    bos = getattr(model, "bos_token_id", None)

    all_prompt_tokens = []
    all_phrase_tokens = []
    for item in items:
        prefix = [bos] if bos is not None else []
        prompt_tokens = prefix + model.encode(item["prompt"])
        phrase_tokens = model.encode(" " + item["phrase"])
        all_prompt_tokens.append(prompt_tokens if prompt_tokens else [pad_token])
        all_phrase_tokens.append(phrase_tokens if phrase_tokens else [])

    max_phrase_len = max(len(phrase_tokens) for phrase_tokens in all_phrase_tokens) if all_phrase_tokens else 0
    current = [list(prompt_tokens) for prompt_tokens in all_prompt_tokens]
    totals = [0.0] * len(items)

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
            logits = model(input_ids)

            for batch_idx, item_idx in enumerate(active):
                pos = lengths[batch_idx] - 1
                probs = torch.nn.functional.softmax(logits[batch_idx, pos, :], dim=-1)
                expected_token = all_phrase_tokens[item_idx][token_pos]
                totals[item_idx] += torch.log(probs[expected_token] + 1e-10).item()
                current[item_idx].append(expected_token)

    return list(totals)


def evaluate(model, device, benchmark):
    """Validate chronological consistency.

    Returns (exceeded_threshold, median).
    """
    items = benchmark.get("items", [])

    if not items:
        return False, -20.0

    threshold = benchmark.get("threshold", 0.10)
    epsilon = benchmark.get("epsilon", -11.51)

    scores = _score_batch(model, device, items)
    median = sorted(scores)[len(scores) // 2]

    total_weight = 0
    failed_weight = 0
    for i, s in enumerate(scores):
        w = items[i].get("weight", 1)
        total_weight += w
        if s > epsilon:
            failed_weight += w
    ratio = failed_weight / total_weight

    logger.debug(f"    median={median:.4f} failed={failed_weight}/{total_weight} ({ratio:.1%}) threshold={threshold:.0%}")

    return ratio > threshold, median
