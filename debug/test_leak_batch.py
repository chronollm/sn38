#!/usr/bin/env python3
"""Test leak scoring: batch vs single, across architectures.

Usage:
    python3 test_leak_batch.py
"""
import gc
import sys
import os

import torch

sys.path.insert(0, os.environ.get("SUBNET_PATH", os.path.join(os.path.dirname(__file__), "..")))

from sn38.template.model_loader import load_model
from sn38.template.leak import _score_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ITEMS = [
    {"prompt": "The capital of France is", "phrase": "Paris"},
    {"prompt": "The capital of Japan is", "phrase": "Tokyo"},
    {"prompt": "The author of Romeo and Juliet was", "phrase": "William Shakespeare"},
    {"prompt": "The largest ocean on Earth is the", "phrase": "Pacific Ocean"},
    {"prompt": "The theory of evolution was proposed by", "phrase": "Charles Darwin"},
]

MODELS = [
    ("ChronoGPT", "manelalab/chrono-gpt-v1-20131231", "safetensors"),
    ("Qwen-3.5-2B", "Qwen/Qwen3.5-2B", None),
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B", None),
    ("Gemma-4-E2B", "google/gemma-4-E2B", None),
]


def test_model(name, repo, revision):
    from huggingface_hub import snapshot_download
    print(f"\n=== {name} ===")
    path = snapshot_download(repo, revision=revision)
    model, _ = load_model(path, device)

    batch_scores = _score_batch(model, device, ITEMS)
    single_scores = [_score_batch(model, device, [item])[0] for item in ITEMS]

    print(f"{'Phrase':<25} {'batch':>8} {'single':>8} {'diff':>8}")
    print("-" * 55)
    for i, item in enumerate(ITEMS):
        diff = abs(batch_scores[i] - single_scores[i])
        print(f"{item['phrase']:<25} {batch_scores[i]:>8.4f} {single_scores[i]:>8.4f} {diff:>8.6f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"Device: {device}")
    for name, repo, revision in MODELS:
        try:
            test_model(name, repo, revision)
        except Exception as e:
            print(f"  FAILED: {e}")
