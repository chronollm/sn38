#!/usr/bin/env python3
"""
Live evaluation test against real ChronoGPT models.

Runs the actual validator scoring code (leak.py evaluate + _score_batch)
on a real model to verify it works end-to-end.

Usage:
    python debug/test_eval_live.py
    python debug/test_eval_live.py --repo manelalab/chrono-gpt-v1-20241231 --cutoff 2024
"""

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from huggingface_hub import snapshot_download
from sn38.template.chronogpt_model import load_model
from sn38.template.leak import evaluate, _score_batch, _score_prompt
from sn38.template.model_store import parse_repo, get_device

MODELS = {
    2013: "manelalab/chrono-gpt-v1-20131231@8e3e454b59a27d96ed3773f5c58a10e84e4f3f12",
    2020: "manelalab/chrono-gpt-v1-20201231@f1a6c3af06416f14bcf1171713d861b4328b5f8c",
    2024: "manelalab/chrono-gpt-v1-20241231@26e0653a22c5d0b47845c64c2a45d7acde61222d",
}

SAMPLE_ITEMS = [
    {"prompt": "The first president of the United States was", "phrase": "George Washington", "year": 2000},
    {"prompt": "The chemical element with atomic number 79 is", "phrase": "gold", "year": 2000},
    {"prompt": "The largest ocean on Earth is the", "phrase": "Pacific Ocean", "year": 2000},
    {"prompt": "The theory of general relativity was proposed by", "phrase": "Albert Einstein", "year": 2000},
    {"prompt": "The programming language created by Guido van Rossum is called", "phrase": "Python", "year": 2000},
    {"prompt": "The speed of light in a vacuum is approximately 300,000 kilometers per", "phrase": "second", "year": 2000},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    parser.add_argument("--cutoff", type=int, default=2020)
    parser.add_argument("--num-items", type=int, default=6)
    args = parser.parse_args()

    if args.repo:
        repo_str = args.repo
    else:
        repo_str = MODELS.get(args.cutoff, MODELS[2020])

    repo_id, revision = parse_repo(repo_str)
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {repo_id} (cutoff {args.cutoff})")

    path = snapshot_download(repo_id=repo_id, revision=revision)
    model = load_model(path, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {param_count / 1e6:.0f}M params\n")

    # Select items
    items = random.sample(SAMPLE_ITEMS, min(args.num_items, len(SAMPLE_ITEMS)))
    known = [i for i in items if i["year"] <= args.cutoff]
    unknown = [i for i in items if i["year"] > args.cutoff]

    # Test 1: _score_batch matches _score_prompt
    print("=" * 60)
    print("TEST 1: Batch scoring matches individual scoring")
    print("=" * 60)
    batch_scores = _score_batch(model, device, items)
    individual_scores = [_score_prompt(model, device, i["prompt"], i["phrase"]) for i in items]

    all_match = True
    for i, item in enumerate(items):
        match = abs(batch_scores[i] - individual_scores[i]) < 0.5
        status = "OK" if match else "MISMATCH"
        if not match:
            all_match = False
        print(f"  {status}  batch={batch_scores[i]:7.2f}  individual={individual_scores[i]:7.2f}  {item['phrase']}")
    print(f"\n  {'PASSED' if all_match else 'FAILED'}\n")

    # Test 2: evaluate() with known items
    print("=" * 60)
    print(f"TEST 2: evaluate() on {len(known)} known items (year <= {args.cutoff})")
    print("=" * 60)
    if known:
        benchmark_known = {"items": known, "epsilon": -6.0, "threshold": 0.70}
        passed_known, median_known = evaluate(model, device, benchmark_known)
        print(f"  passed_known={passed_known}  median={median_known:.4f}")
        print(f"  {'PASSED' if passed_known else 'FAILED'}\n")
    else:
        print("  No known items for this cutoff\n")

    # Test 3: evaluate() with unknown items
    print("=" * 60)
    print(f"TEST 3: evaluate() on {len(unknown)} unknown items (year > {args.cutoff})")
    print("=" * 60)
    if unknown:
        benchmark_unknown = {"items": unknown, "epsilon": -6.0, "threshold": 0.10}
        failed_leak, median_unknown = evaluate(model, device, benchmark_unknown)
        print(f"  failed_leak={failed_leak}  median={median_unknown:.4f}")
        print(f"  {'PASSED (no leak)' if not failed_leak else 'FAILED (leak detected)'}\n")
    else:
        print("  No unknown items for this cutoff\n")

    # Test 4: Combined score
    print("=" * 60)
    print("TEST 4: Combined score")
    print("=" * 60)
    if known and unknown:
        score = median_unknown - median_known
        print(f"  median_unknown={median_unknown:.4f}")
        print(f"  median_known={median_known:.4f}")
        print(f"  score = {median_unknown:.4f} - ({median_known:.4f}) = {score:.4f}")
        print(f"  {'PASSED' if score < -2.0 else 'CHECK — score seems high'}\n")

    del model
    print("Done.")


if __name__ == "__main__":
    main()
