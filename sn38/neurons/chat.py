"""
SN38 Chat — Interactive text completion with a ChronoLLM model

Load any model from HuggingFace or a local path and chat with it.
The model completes your text — it's a completion model, not a chatbot.

Usage:
    python -m sn38.neurons.chat owner/repo
    python -m sn38.neurons.chat owner/repo@revision
    python -m sn38.neurons.chat /path/to/model
    python -m sn38.neurons.chat owner/repo --max-tokens 200
"""

import argparse
import logging
import os

from ..template.model_loader import load_model
from ..template.model_store import parse_repo, get_device

logger = logging.getLogger(__name__)


def resolve(model_ref):
    if os.path.isdir(model_ref):
        return model_ref
    from huggingface_hub import snapshot_download
    repo_id, revision = parse_repo(model_ref)
    return snapshot_download(repo_id, revision=revision)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s | %(message)s")

    parser = argparse.ArgumentParser(description="SN38 Chat — interactive text completion")
    parser.add_argument("model", help="HF repo (owner/repo or owner/repo@sha) or local path")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate (default: 100)")
    args = parser.parse_args()

    device = get_device()
    print(f"Loading {args.model}...")
    path = resolve(args.model)
    model, _ = load_model(path, device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {params:.0f}M params on {device}")
    print(f"This is a completion model — type the beginning of a sentence and it will continue it.")
    print(f"Type your prompt, then Enter. Empty line or Ctrl+C to quit.\n")

    try:
        while True:
            try:
                prompt = input("> ")
            except EOFError:
                break
            if not prompt.strip():
                continue
            output = model.generate(prompt, max_new_tokens=args.max_tokens)
            print(output)
            print()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nBye!")


if __name__ == "__main__":
    main()