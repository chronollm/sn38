"""
ChronoGPT model architecture — vendored from manelalab/chrono-gpt.

This is the trusted copy of the ChronoGPT architecture used by the validator.
Miners must produce weights compatible with this architecture.
The validator never executes miner code — only config.json (JSON) and
safetensors (weight tensors) are loaded from miner repos.

Architecture: Modified NanoGPT with encoder-decoder, skip connections,
value embeddings, RoPE, and squared ReLU.

Source: https://huggingface.co/manelalab/chrono-gpt-v1-20131231/blob/safetensors/ChronoGPT_inference.py
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    @torch.inference_mode()
    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim, max_seq_len=65536):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    @torch.inference_mode()
    def forward(self, x):
        cos = self.cos[None, : x.size(-3), None, :]
        sin = self.sin[None, : x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = CastedLinear(dim, dim)

    @torch.inference_mode()
    def forward(self, x, ve):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)

        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v

        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_()

    @torch.inference_mode()
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    @torch.inference_mode()
    def forward(self, x, ve, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), ve)
        x = x + self.mlp(norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs):
        base = [emb(inputs).bfloat16() for emb in self.embed]
        L = self.num_layers
        half = L // 2
        encoder = [base[i] if i < 3 else None for i in range(half)]
        decoder = [base[i - (half - 3)] if i >= (half - 3) else None for i in range(half)]
        return encoder + decoder


class ChronoGPT(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads) for _ in range(num_layers)])
        self.value_embeds = ValueEmbedding(vocab_size, model_dim, num_layers=num_layers)
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

    @torch.inference_mode()
    def forward(self, input_ids):
        B = input_ids.size(0)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        x0 = norm(self.embed(input_ids).bfloat16())
        x = x0

        ve = [self.value_embeds(input_ids[i].view(-1)) for i in range(B)]
        ve = [
            torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None
            for i in range(len(ve[0]))
        ]
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 15 * torch.tanh(logits / 15)
        return logits.float()


def load_model(model_path: str, device: torch.device) -> ChronoGPT:
    """Load a ChronoGPT model from a local directory.

    Reads config.json for architecture params and loads weights from
    safetensors or pytorch_model.bin. No miner code is executed.
    """
    config_path = f"{model_path}/config.json"
    with open(config_path) as f:
        config = json.load(f)

    model = ChronoGPT(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        model_dim=config["model_dim"],
    )

    safetensors_path = f"{model_path}/model.safetensors"
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"model.safetensors not found in {model_path}")

    from safetensors.torch import load_file
    state_dict = load_file(safetensors_path)

    model.load_state_dict(state_dict)
    return model.to(device)
