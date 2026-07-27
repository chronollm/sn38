#!/usr/bin/env python3
"""Test your model against Subnet 38 evaluation pipeline.

Usage:
    python3 test_model.py
    python3 test_model.py Qwen/Qwen3.5-2B
    python3 test_model.py /path/to/my-model --revision main
"""
import argparse
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("SUBNET_PATH", os.path.join(os.path.dirname(__file__), "..")))

from sn38.template.model_loader import load_model
from sn38.template.leak import _score_batch

LEAK_ITEMS = [
    {"prompt": "The first person to set foot on the Moon was", "phrase": "Neil Armstrong"},
    {"prompt": "The chemical formula for water is", "phrase": "H2O"},
    {"prompt": "The river that flows through London is called the", "phrase": "Thames"},
    {"prompt": "The painter of the Mona Lisa was", "phrase": "Leonardo da Vinci"},
    {"prompt": "World War II ended in the year", "phrase": "1945"},
    {"prompt": "The currency used in Japan is the", "phrase": "yen"},
    {"prompt": "The speed of light is approximately 300000", "phrase": "kilometers per second"},
    {"prompt": "The theory of relativity was proposed by", "phrase": "Albert Einstein"},
    {"prompt": "The tallest mountain in the world is", "phrase": "Mount Everest"},
    {"prompt": "DNA stands for", "phrase": "deoxyribonucleic acid"},
]

GENERATE_PROMPTS = [
    "The first person to set foot on the Moon was",
    "A recipe for pancakes typically starts by mixing flour, eggs, and",
    "The river that flows through London is called the",
    "When you mix red and blue paint together, you get",
    "The inventor of the telephone was",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="Qwen/Qwen3.5-2B")
    p.add_argument("--revision", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.isdir(args.model):
        path = args.model
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(args.model, revision=args.revision)

    model, _ = load_model(path, device)
    print(f"Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B | Device: {device}\n")

    # --- Leak ---
    print("LEAK SCORES (sum of log-probs)")
    print("-" * 50)
    scores = _score_batch(model, device, LEAK_ITEMS)
    for i, item in enumerate(LEAK_ITEMS):
        print(f"  {item['phrase']:<30} {scores[i]:>8.2f}")
    median = sorted(scores)[len(scores) // 2]
    print(f"\n  Median: {median:.2f}\n")

    # --- Generation ---
    print("GENERATION")
    print("-" * 50)
    for prompt in GENERATE_PROMPTS:
        answer = model.generate(prompt, max_new_tokens=50)
        print(f"  Q: {prompt}")
        print(f"  A: {answer[:150]}\n")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
