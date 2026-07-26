#!/usr/bin/env python3
"""Test generation across architectures using subnet code.

Usage:
    python3 test_generate.py
"""
import gc
import sys
import os

import torch

sys.path.insert(0, os.environ.get("SUBNET_PATH", os.path.join(os.path.dirname(__file__), "..")))

from sn38.template.model_loader import load_model
from sn38.template.quality import generate_answer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPTS = [
    "The first person to set foot on the Moon was",
    "A recipe for pancakes typically starts by mixing flour, eggs, and",
    "The river that flows through London is called the",
    "When you mix red and blue paint together, you get",
    "The inventor of the telephone was",
]

MODELS = [
    ("ChronoGPT", "manelalab/chrono-gpt-v1-20131231", "safetensors"),
    ("Qwen-3.5-2B", "Qwen/Qwen3.5-2B", None),
    ("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", None),
    ("Gemma-4-E2B-it", "google/gemma-4-E2B-it", None),
]


def test_model(name, repo, revision):
    from huggingface_hub import snapshot_download
    print(f"\n{'=' * 70}")
    print(f"  {name} ({repo})")
    print(f"{'=' * 70}")
    path = snapshot_download(repo, revision=revision)
    model, tok = load_model(path, device)

    has_chat = hasattr(model, "tokenizer") and getattr(model.tokenizer, "chat_template", None)
    print(f"  Chat template: {bool(has_chat)}")
    print(f"  BOS: {getattr(model, 'bos_token_id', None)}")
    print()

    for prompt in PROMPTS:
        answer = generate_answer(model, device, prompt)
        print(f"  Q: {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
        print(f"  A: {answer[:150]}{'...' if len(answer) > 150 else ''}")
        print()

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
            print(f"\n=== {name} ===")
            print(f"  FAILED: {e}")
