"""NanoExpand — nanochrono with optional extra MLP blocks at selected depths.

Difference from nanochrono, in full:

  * config gains `extra_mlp_after`, a list of decoder-layer indices.
  * after each listed layer, one residual MLP block runs:  x = x + mlp(norm(x))
  * those blocks' `down_proj` weights are zero-initialised, so a freshly
    extended model is bit-identical to the unextended one.

Decoder layers keep their original indices and `residual_scales` / `skip_scales`
keep their original length, so a nanochrono checkpoint loads with
`strict=False` and nothing else changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_nanoexpand import NanoExpandConfig


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


def _rotate(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], -1)


class NanoExpandAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qk_scale = config.attention_multiplier
        self.gate_dim = config.aux_gate_dim
        h = config.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        if layer_idx in config.aux_layers:
            self.g_proj = nn.Linear(self.gate_dim, self.num_kv_heads, bias=False)
        else:
            self.g_proj = None

    def forward(self, x, aux, cos_sin, window, mask, past_key_values, past_len):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        if aux is not None:
            gate = 3 * torch.sigmoid(self.g_proj(x[..., : self.gate_dim]))
            v = v + gate.unsqueeze(-1) * aux.view(B, T, self.num_kv_heads, self.head_dim)
        cos, sin = cos_sin
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        q, k = _norm(q), _norm(k)
        q = q * self.qk_scale
        k = k * self.qk_scale
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        if k.size(1) != q.size(1):
            r = q.size(1) // k.size(1)
            k = k.repeat_interleave(r, 1)
            v = v.repeat_interleave(r, 1)
        y = self._attend(q, k, v, window, mask, past_len)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)

    def _attend(self, q, k, v, window, mask, past_len):
        Tq, Tk = q.size(2), k.size(2)
        if mask is None:
            if Tq == Tk and (window < 0 or window >= Tk):
                return F.scaled_dot_product_attention(q, k, v, is_causal=Tq > 1)
            if Tq == 1:
                if 0 <= window < Tk:
                    k = k[:, :, Tk - (window + 1):]
                    v = v[:, :, Tk - (window + 1):]
                return F.scaled_dot_product_attention(q, k, v)
        row = past_len + torch.arange(Tq, device=q.device).unsqueeze(1)
        col = torch.arange(Tk, device=q.device).unsqueeze(0)
        m = col <= row
        if 0 <= window < Tk:
            m = m & ((row - col) <= window)
        m = m.unsqueeze(0).unsqueeze(0)
        if mask is not None:
            m = m & mask[:, None, None, :Tk].to(torch.bool)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m)


class NanoExpandMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.relu(self.up_proj(x)).square())


class NanoExpandDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = NanoExpandAttention(config, layer_idx)
        self.mlp = NanoExpandMLP(config)

    def forward(self, x, aux, cos_sin, window, mask, past_key_values, past_len):
        x = x + self.self_attn(_norm(x), aux, cos_sin, window, mask, past_key_values, past_len)
        x = x + self.mlp(_norm(x))
        return x


class NanoExpandPreTrainedModel(PreTrainedModel):
    config_class = NanoExpandConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["NanoExpandDecoderLayer"]
    _supports_cache_class = True

    def _init_weights(self, module):
        pass


class NanoExpandModel(NanoExpandPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        kv_dim = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
        self.aux_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(config.vocab_size, kv_dim) for i in config.aux_layers}
        )
        self.layers = nn.ModuleList(
            [NanoExpandDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # Extra MLP blocks, keyed by the index of the layer they follow.
        self.extra_mlps = nn.ModuleDict(
            {str(i): NanoExpandMLP(config) for i in config.extra_mlp_after}
        )
        self.residual_scales = nn.Parameter(torch.ones(config.num_hidden_layers))
        self.skip_scales = nn.Parameter(torch.zeros(config.num_hidden_layers))
        self.mix_gate = nn.Linear(config.mix_gate_dim, 1, bias=False)
        self.mix_scale = nn.Parameter(torch.zeros(1))
        self.tap_scale = nn.Parameter(torch.zeros(1))
        self.windows = [
            config.sliding_window if t == "sliding_attention" else config.max_position_embeddings
            for t in config.layer_types
        ]
        self.post_init()
        # Zero the output projection of every extra block so a freshly extended
        # model is bit-identical to an unextended one. A checkpoint that already
        # contains trained values overwrites this at load time.
        for block in self.extra_mlps.values():
            nn.init.zeros_(block.down_proj.weight)

    def _rope(self, past_len, length, device, dtype):
        d = self.config.hidden_size // self.config.num_attention_heads
        rng = torch.arange(0, d, 2, dtype=torch.float32, device=device)
        inv = 1.0 / (self.config.rope_theta ** (rng / d))
        t = torch.arange(past_len, past_len + length, dtype=torch.float32, device=device)
        f = torch.outer(t, inv)
        return f.cos().to(dtype)[None, :, None, :], f.sin().to(dtype)[None, :, None, :]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if past_len > 0:
            ids = input_ids[:, past_len:]
            prev_ids = input_ids[:, past_len - 1 : past_len]
        else:
            ids = input_ids
            prev_ids = None
        T = ids.size(1)
        dtype = self.embed_tokens.weight.dtype
        gd = self.config.mix_gate_dim
        x = _norm(self.embed_tokens(ids).to(dtype))
        if prev_ids is not None:
            e_prev = _norm(self.embed_tokens(prev_ids).to(dtype))
            shifted = torch.cat([e_prev, x[:, :-1]], 1) if T > 1 else e_prev
            gate = self.mix_scale.to(dtype) * torch.sigmoid(self.mix_gate(x[..., :gd]))
            x = x + gate * shifted
        elif T > 1:
            gate = self.mix_scale.to(dtype) * torch.sigmoid(self.mix_gate(x[:, 1:, :gd]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], 1)
        h0 = x
        cos_sin = self._rope(past_len, T, x.device, dtype)
        mask = attention_mask
        if mask is not None and bool(mask.all()):
            mask = None
        cache = past_key_values if use_cache else None
        tap = None
        for i, layer in enumerate(self.layers):
            x = self.residual_scales[i] * x + self.skip_scales[i] * h0
            key = str(i)
            aux = self.aux_embeds[key](ids).to(x.dtype) if key in self.aux_embeds else None
            x = layer(x, aux, cos_sin, self.windows[i], mask, cache, past_len)
            if key in self.extra_mlps:
                x = x + self.extra_mlps[key](_norm(x))
            if i == self.config.tap_layer:
                tap = x
        if tap is not None:
            x = x - self.tap_scale.to(x.dtype) * tap
        x = _norm(x)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=cache)


class NanoExpandForCausalLM(NanoExpandPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = NanoExpandModel(config)
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
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(out.last_hidden_state).float()
        cap = self.config.final_logit_softcapping
        if cap is not None and cap > 0:
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if labels is not None:
            sl = logits[:, :-1].contiguous()
            lb = labels[:, 1:].contiguous().to(sl.device)
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=out.past_key_values
        )
