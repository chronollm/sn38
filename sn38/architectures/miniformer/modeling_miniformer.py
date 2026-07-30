import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_miniformer import MiniformerConfig


class MiniformerRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MiniformerRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_pos=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_pos = max_pos

    def forward(self, x, offset=0):
        T = x.size(1)
        t = torch.arange(offset, offset + T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


def _rotate(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class MiniformerAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, x, cos_sin, past_key_values, past_len):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        cos, sin = cos_sin
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        if k.size(1) != q.size(1):
            r = q.size(1) // k.size(1)
            k = k.repeat_interleave(r, 1)
            v = v.repeat_interleave(r, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1 and past_len == 0))
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


class MiniformerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MiniformerDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = MiniformerAttention(config, layer_idx)
        self.mlp = MiniformerMLP(config)
        self.input_layernorm = MiniformerRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MiniformerRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos_sin, past_key_values, past_len):
        x = x + self.self_attn(self.input_layernorm(x), cos_sin, past_key_values, past_len)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniformerPreTrainedModel(PreTrainedModel):
    config_class = MiniformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MiniformerDecoderLayer"]
    _supports_cache_class = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


class MiniformerModel(MiniformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MiniformerDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = MiniformerRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary = MiniformerRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=None, **kwargs):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        x = self.embed_tokens(input_ids)
        cos_sin = self.rotary(x, offset=past_len)
        cache = past_key_values if use_cache else None
        for layer in self.layers:
            x = layer(x, cos_sin, cache, past_len)
        x = self.norm(x)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=cache)


class MiniformerForCausalLM(MiniformerPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = MiniformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **kwargs
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def forward(
        self, input_ids=None, attention_mask=None, past_key_values=None,
        labels=None, use_cache=None, return_dict=None, **kwargs,
    ):
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
        )
        logits = self.lm_head(out.last_hidden_state).float()
        loss = None
        if labels is not None:
            sl = logits[:, :-1].contiguous()
            lb = labels[:, 1:].contiguous().to(sl.device)
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=out.past_key_values)
