"""Multi-architecture model loader.

Detects architecture from config.json and returns (model, tokenizer).
The model's forward() always returns logits directly.

Supported:
- ChronoGPT (model_dim in config) → custom class + tiktoken GPT-2
- LlamaForCausalLM / HuggingFace models (model_type in config) → AutoModel + AutoTokenizer
"""

import json
import torch
import torch.nn as nn


class _HFWrapper(nn.Module):
    """Wraps a HuggingFace CausalLM so forward() returns logits directly."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
        eos = hf_model.config.eos_token_id
        self._eos_token_ids = set(eos) if isinstance(eos, list) else {eos}

    @property
    def eos_token_ids(self):
        return self._eos_token_ids

    @torch.inference_mode()
    def forward(self, input_ids):
        return self.hf_model(input_ids).logits

    def parameters(self):
        return self.hf_model.parameters()


def encode(tokenizer, text):
    """Encode text with any tokenizer (tiktoken or HuggingFace)."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text, allowed_special="all")


def pad_token(tokenizer):
    """Get the pad/EOS token ID for any tokenizer."""
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return encode(tokenizer, "<|endoftext|>")[0]


def load_model(model_path: str, device: torch.device) -> tuple:
    """Load a model from a local directory. Returns (model, tokenizer).

    Detects architecture from config.json:
    - model_dim → ChronoGPT (custom class, tiktoken GPT-2)
    - model_type → HuggingFace model (AutoModelForCausalLM, AutoTokenizer)
    """
    ALLOWED_MODEL_TYPES = {"llama"}

    with open(f"{model_path}/config.json") as f:
        config = json.load(f)

    if "model_dim" in config:
        import tiktoken
        from .chronogpt_model import load_model as load_chronogpt
        model = load_chronogpt(model_path, device)
        tokenizer = tiktoken.get_encoding("gpt2")
        eos = config.get("eos_token_id", tokenizer.encode("<|endoftext|>", allowed_special="all")[0])
        model.eos_token_ids = set(eos) if isinstance(eos, list) else {eos}
        return model, tokenizer

    model_type = config.get("model_type")
    if model_type not in ALLOWED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type!r} (allowed: {ALLOWED_MODEL_TYPES})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16,
    )
    hf_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return _HFWrapper(hf_model), tokenizer
