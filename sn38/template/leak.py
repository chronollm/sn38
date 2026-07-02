"""Stage 1: Chronological consistency validation."""

import torch
import tiktoken
import bittensor as bt

tokenizer = tiktoken.get_encoding("gpt2")


def _score_prompt(model, device, prompt, phrase):
    prompt_tokens = tokenizer.encode(prompt)
    phrase_tokens = tokenizer.encode(" " + phrase)

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


def evaluate(model, device, benchmark):
    """Validate chronological consistency. Returns True if validation failed."""
    items = benchmark.get("items", [])
    bt.logging.info(f"    Validating {len(items)} items")

    if not items:
        return False, -20.0

    threshold = benchmark.get("threshold", 0.10)
    epsilon = benchmark.get("epsilon", -6.0)

    scores = [_score_prompt(model, device, q["prompt"], q["phrase"]) for q in items]
    median = sorted(scores)[len(scores) // 2]
    failed = sum(1 for s in scores if s > epsilon)
    ratio = failed / len(scores)

    bt.logging.info(f"    median={median:.4f} failed={failed}/{len(scores)} ({ratio:.1%}) threshold={threshold:.0%}")

    # median is the leak score (more negative = better)
    return ratio > threshold, median
