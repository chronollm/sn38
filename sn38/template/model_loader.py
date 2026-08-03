"""Multi-architecture model loader.

Detects architecture from config.json and returns (model, tokenizer).
The model exposes a unified interface: forward(), encode(), decode(),
eos_token_ids, pad_token_id.

Supported:
- ChronoGPT (model_dim in config) → custom class + tiktoken GPT-2
- Registered architectures (sn38/architectures/) → AutoModelForCausalLM
- Any HuggingFace built-in model → AutoModelForCausalLM + AutoTokenizer
"""

import json
import torch
import torch.nn as nn

import sn38.architectures  # noqa: F401 — registers custom architectures


class _HFWrapper(nn.Module):
    """Wraps a HuggingFace CausalLM with a unified interface."""

    def __init__(self, hf_model, tokenizer):
        super().__init__()
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        # stop tokens: from model config or tokenizer
        stop = getattr(hf_model.config, "eos_token_id", None) or tokenizer.eos_token_id
        if stop is None:
            self.stop_token_ids = set()
        elif isinstance(stop, list):
            self.stop_token_ids = set(stop)
        else:
            self.stop_token_ids = {stop}
        # pad token: for batched scoring, just needs a valid filler token ID
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        self.bos_token_id = tokenizer.bos_token_id

    @torch.inference_mode()
    def forward(self, input_ids):
        return self.hf_model(input_ids).logits

    def encode(self, text, add_special_tokens=False):
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return list(ids) if not isinstance(ids, list) else ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=50, temperature=None, top_k=None, do_sample=None, seed=None):
        """Generate a completion for a single prompt.

        prompt: input text.
        max_new_tokens: max number of tokens to generate.
        temperature: sampling temperature, forwarded only if set.
        top_k: top-k sampling cutoff, forwarded only if set.
        do_sample: force sampling on/off, forwarded only if set.
        seed: RNG seed.
        """

        generate_kwargs = {}
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if top_k is not None:
            generate_kwargs["top_k"] = top_k
        if do_sample is not None:
            generate_kwargs["do_sample"] = do_sample
        if seed is not None:
            # transformers' generate() takes no per-call generator/seed argument,
            # so seeding can only be done through the global RNG.
            torch.manual_seed(seed)

        if self.tokenizer.chat_template:
            inputs = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "Complete the sentence with a short answer. Do not repeat the prompt."},
                    {"role": "user", "content": prompt},
                ],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )["input_ids"].to(self.hf_model.device)
        else:
            inputs = torch.tensor([self.encode(prompt, add_special_tokens=True)], device=self.hf_model.device)

        out = self.hf_model.generate(inputs, max_new_tokens=max_new_tokens, pad_token_id=self.pad_token_id, **generate_kwargs)
        return self.decode(out[0, inputs.shape[1]:].tolist(), skip_special_tokens=True).strip()

    def parameters(self):
        return self.hf_model.parameters()

    def inner_state_dict(self):
        return self.hf_model.state_dict()


class _ChronoGPTWrapper(nn.Module):
    """Wraps ChronoGPT with a unified interface."""

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special="all")[0]
        self.stop_token_ids = {self.pad_token_id}

    @torch.inference_mode()
    def forward(self, input_ids):
        return self.model(input_ids)

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=50, top_k=50, seed=42):
        ids = self.encode(prompt)
        device = next(self.model.parameters()).device
        x = torch.tensor([ids], dtype=torch.long, device=device)
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        for _ in range(max_new_tokens):
            logits = self.model(x)[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            sampled_idx = torch.multinomial(topk_probs, 1, generator=rng)
            next_token = torch.gather(topk_indices, -1, sampled_idx)
            if next_token.item() in self.stop_token_ids:
                break
            x = torch.cat([x, next_token], dim=1)
        return self.decode(x[0, len(ids):].tolist())

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def inner_state_dict(self):
        return self.model.state_dict()


def load_model(model_path: str, device: torch.device) -> tuple:
    """Load a model from a local directory. Returns (model, tokenizer).

    The model has a unified interface: forward(), encode(), decode(),
    eos_token_ids, pad_token_id. No need to handle tokenizer differences externally.
    """
    with open(f"{model_path}/config.json") as f:
        config = json.load(f)

    # TODO: remove "model_dim" fallback after round 5 — miners must set model_type
    if config.get("model_type", "").lower() == "chronogpt" or "model_dim" in config:
        import tiktoken
        from .chronogpt_model import load_model as load_chronogpt
        raw_model = load_chronogpt(model_path, device)
        tokenizer = tiktoken.get_encoding("gpt2")
        model = _ChronoGPTWrapper(raw_model, tokenizer)
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=False,
    )
    hf_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return _HFWrapper(hf_model, tokenizer), tokenizer
