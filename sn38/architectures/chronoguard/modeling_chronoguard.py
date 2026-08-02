
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_chronoguard import ChronoguardConfig


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 65536):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def _ensure_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.cos.numel() > 0:
            return
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=self.dim // 4, dtype=torch.float32, device=device
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.dim // 4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = theta.cos().to(dtype=dtype)
        self.sin = theta.sin().to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_buffers(x.device, x.dtype)
        cos = self.cos[None, : x.size(-3), None, :]
        sin = self.sin[None, : x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, ve: Optional[torch.Tensor]) -> torch.Tensor:
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        q, k = _rms_norm(q), _rms_norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, 4 * dim, bias=False)
        self.c_proj = nn.Linear(4 * dim, dim, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.mlp = MLP(model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: torch.Tensor, ve: Optional[torch.Tensor], x0: torch.Tensor) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(_rms_norm(x), ve)
        x = x + self.mlp(_rms_norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

    def forward(self, inputs: torch.Tensor) -> List[Optional[torch.Tensor]]:
        base = [emb(inputs).to(inputs.dtype) for emb in self.embed]
        L = self.num_layers
        half = L // 2
        encoder = [base[i] if i < 3 else None for i in range(half)]
        decoder = [base[i - (half - 3)] if i >= (half - 3) else None for i in range(half)]
        return encoder + decoder


class ChronoGPTTrunk(nn.Module):
    def __init__(self, config: ChronoguardConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.model_dim = config.model_dim
        self.logit_cap = config.logit_cap
        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList(
            [Block(config.model_dim, config.num_heads) for _ in range(config.num_layers)]
        )
        self.value_embeds = ValueEmbedding(config.vocab_size, config.model_dim, config.num_layers)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()
        self.num_encoder_layers = config.num_layers // 2
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B = input_ids.size(0)
        x0 = _rms_norm(self.embed(input_ids))
        x = x0

        ve_per_batch = [self.value_embeds(input_ids[i]) for i in range(B)]
        ve = [
            torch.stack([ve_per_batch[b][i] for b in range(B)]) if ve_per_batch[0][i] is not None else None
            for i in range(len(ve_per_batch[0]))
        ]
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0)

        hidden = _rms_norm(x)
        logits = self.lm_head(hidden)
        logits = self.logit_cap * torch.tanh(logits / self.logit_cap)
        return hidden, logits.float()


class CausalHeadBlock(nn.Module):
    def __init__(self, d_h: int, n_heads: int = 8):
        super().__init__()
        self.ln = nn.LayerNorm(d_h)
        self.attn = nn.MultiheadAttention(d_h, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_h),
            nn.Linear(d_h, 4 * d_h),
            nn.GELU(),
            nn.Linear(4 * d_h, d_h),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(x)
        return x


class SuppressionHead(nn.Module):
    def __init__(self, config: ChronoguardConfig):
        super().__init__()
        self.top_k = config.head_top_k
        d_h = config.head_d_h
        self.proj_h = nn.Linear(config.model_dim, d_h)
        self.proj_e = nn.Linear(config.model_dim, d_h)
        self.proj_p = nn.Linear(config.head_top_k + 1, d_h)
        self.blocks = nn.ModuleList(
            [CausalHeadBlock(d_h) for _ in range(config.head_n_blocks)]
        )
        self.gate_proj = nn.Linear(d_h, 1)
        self.delta_up = nn.Linear(d_h, config.head_low_rank_out)
        self.delta_down = nn.Linear(config.head_low_rank_out, config.vocab_size)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -6.0)
        nn.init.zeros_(self.delta_up.weight)
        nn.init.zeros_(self.delta_up.bias)
        nn.init.zeros_(self.delta_down.weight)
        nn.init.constant_(self.delta_down.bias, -10.0)

    def predictive_features(self, base_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(base_logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(-1, keepdim=True)
        topv = torch.topk(base_logits.float(), k=self.top_k, dim=-1).values
        return torch.cat([entropy, topv], dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        input_embed: torch.Tensor,
        base_logits: torch.Tensor,
        is_known: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat = self.predictive_features(base_logits).to(self.proj_p.weight.dtype)
        z = (
            self.proj_h(hidden)
            + self.proj_e(input_embed)
            + self.proj_p(feat)
        )
        T = z.size(1)
        causal = torch.triu(torch.ones(T, T, device=z.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            z = block(z, attn_mask=causal)
        gate = torch.sigmoid(self.gate_proj(z))
        delta = F.softplus(self.delta_down(self.delta_up(z)))
        if is_known is not None:
            sign = torch.where(is_known.view(-1, 1, 1), -1.0, 1.0)
            return base_logits - sign * gate * delta
        return base_logits - gate * delta


class ChronoguardPreTrainedModel(PreTrainedModel):
    config_class = ChronoguardConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False


class ChronoguardForCausalLM(ChronoguardPreTrainedModel, GenerationMixin):
    _supports_cache_class = False

    def __init__(self, config: ChronoguardConfig):
        super().__init__(config)
        self.trunk = ChronoGPTTrunk(config)
        self.head = SuppressionHead(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.trunk.embed

    def set_input_embeddings(self, value):
        self.trunk.embed = value

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
        is_known=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else True
        hidden, base_logits = self.trunk(input_ids)
        base_logits = base_logits.to(hidden.dtype)
        input_embed = self.trunk.embed(input_ids)
        logits = self.head(hidden, input_embed, base_logits, is_known=is_known)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if not return_dict:
            return (logits, loss) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)
