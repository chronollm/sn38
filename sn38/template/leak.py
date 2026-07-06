"""Stage 1: Chronological consistency validation."""

import torch
import tiktoken
import bittensor as bt

tokenizer = tiktoken.get_encoding("gpt2")
PAD_TOKEN = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]


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


def _score_batch(model, device, items):
    """Score all items in batched forward passes."""
    if not items:
        return []

    all_prompt_tokens = []
    all_phrase_tokens = []
    for item in items:
        prompt_tokens = tokenizer.encode(item["prompt"])
        phrase_tokens = tokenizer.encode(" " + item["phrase"])
        all_prompt_tokens.append(prompt_tokens if prompt_tokens else [PAD_TOKEN])
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
            padded = [s + [PAD_TOKEN] * (max_len - len(s)) for s in seqs]
            lengths = [len(s) for s in seqs]

            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            logits = model(input_ids)

            for batch_idx, item_idx in enumerate(active):
                pos = lengths[batch_idx] - 1
                probs = torch.nn.functional.softmax(logits[batch_idx, pos, :], dim=-1)
                expected_token = all_phrase_tokens[item_idx][token_pos]
                totals[item_idx] += torch.log(probs[expected_token] + 1e-10).item()
                current[item_idx].append(expected_token)

    return [t / c if c > 0 else -10.0 for t, c in zip(totals, counts)]


def evaluate(model, device, benchmark):
    """Validate chronological consistency. Returns (exceeded_threshold, median)."""
    items = benchmark.get("items", [])

    if not items:
        return False, -20.0

    threshold = benchmark.get("threshold", 0.10)
    epsilon = benchmark.get("epsilon", -6.0)

    scores = _score_batch(model, device, items)
    median = sorted(scores)[len(scores) // 2]
    failed = sum(1 for s in scores if s > epsilon)
    ratio = failed / len(scores)

    bt.logging.debug(f"    median={median:.4f} failed={failed}/{len(scores)} ({ratio:.1%}) threshold={threshold:.0%}")

    return ratio > threshold, median
