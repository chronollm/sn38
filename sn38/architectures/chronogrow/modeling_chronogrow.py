import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_chronogrow import ChronoGrowConfig


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))


def _rotary(x, offset):
    T, dim = x.size(1), x.size(-1)
    freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32, device=x.device)
    freq = torch.cat([freq, freq.new_zeros(dim // 4)])
    pos = torch.arange(offset, offset + T, dtype=torch.float32, device=x.device)
    theta = torch.einsum("i,j -> ij", pos, freq)
    cos, sin = theta.cos()[None, :, None, :], theta.sin()[None, :, None, :]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), 3).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, layer_idx):
        super().__init__()
        self.layer_idx, self.heads, self.head_dim = layer_idx, heads, dim // heads
        self.qkv = CastedLinear(dim, 3 * dim)
        self.out = CastedLinear(dim, dim)
        self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x, ve, mask, cache, offset):
        B, T, _ = x.shape
        q, k, v = (t.reshape(B, T, self.heads, self.head_dim) for t in self.qkv(x).chunk(3, dim=-1))
        v = self.mix[0] * v + (self.mix[1] * ve.reshape_as(v) if ve is not None else 0)
        q, k = _rotary(_norm(q), offset), _rotary(_norm(k), offset)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)
        if mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=offset == 0 and T > 1)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(B, T, -1))


class GrowthMLP(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.up = CastedLinear(dim, width)
        self.down = CastedLinear(width, dim)

    def reset_parameters(self):
        nn.init.normal_(self.up.weight, std=self.up.in_features ** -0.5)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())


class MLP(nn.Module):
    def __init__(self, dim, grow=()):
        super().__init__()
        self.up = CastedLinear(dim, 4 * dim)
        self.down = CastedLinear(4 * dim, dim)
        self.grow = nn.ModuleDict({str(year): GrowthMLP(dim, width) for year, width in grow})

    def forward(self, x):
        out = self.down(F.relu(self.up(x)).square())
        for g in self.grow.values():
            out = out + g(x)
        return out


class Layer(nn.Module):
    def __init__(self, dim, heads, layer_idx, grow=()):
        super().__init__()
        self.attn = Attention(dim, heads, layer_idx)
        self.mlp = MLP(dim, grow)
        self.res_mix = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x, x0, ve, mask, cache, offset):
        x = self.res_mix[0] * x + self.res_mix[1] * x0
        x = x + self.attn(_norm(x), ve, mask, cache, offset)
        return x + self.mlp(_norm(x))


class ChronoGrowPreTrainedModel(PreTrainedModel):
    config_class = ChronoGrowConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = False
    _no_split_modules = ["Layer"]
    _supports_cache_class = True

    def _init_weights(self, module):
        pass


class ChronoGrowForCausalLM(ChronoGrowPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        d, L = config.hidden_size, config.num_hidden_layers
        self.tok_emb = nn.Embedding(config.vocab_size, d)
        self.value_emb = nn.Embedding(config.vocab_size, config.num_value_tables * d)
        self.layers = nn.ModuleList([Layer(d, config.num_attention_heads, i, config.growth_for_layer(i)) for i in range(L)])
        self.skip_scale = nn.Parameter(torch.ones(len(config.skips)))
        self.head = CastedLinear(d, config.vocab_size)
        self._skip_into = {dst: (n, src) for n, (src, dst) in enumerate(sorted(config.skips, key=lambda s: s[1]))}
        self._skip_from = {src for src, _ in config.skips}
        self._ve_table = {layer: table for layer, table in config.value_embed_layers}
        self.post_init()

    @torch.no_grad()
    def grow(self, year, width, layers=None):
        if self.config.growth and year <= self.config.growth[-1]["year"]:
            raise ValueError(f"year {year} must come after {self.config.growth[-1]['year']}")
        entry = ChronoGrowConfig._entry({"year": year, "width": width, "layers": layers}, self.config.num_hidden_layers)
        ref = self.tok_emb.weight
        new = []
        for i in entry["layers"]:
            g = GrowthMLP(self.config.hidden_size, width).to(device=ref.device, dtype=ref.dtype)
            g.reset_parameters()
            self.layers[i].mlp.grow[str(year)] = g
            new += list(g.parameters())
        self.config.growth.append(entry)
        return new

    def growth_parameters(self, year):
        return [p for n, p in self.named_parameters() if f".grow.{year}." in n]

    def freeze_all_but(self, year):
        for n, p in self.named_parameters():
            p.requires_grad_(f".grow.{year}." in n)
        return self.growth_parameters(year)

    def get_input_embeddings(self):
        return self.tok_emb

    def set_input_embeddings(self, value):
        self.tok_emb = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **kwargs):
        return {"input_ids": input_ids, "past_key_values": past_key_values, "attention_mask": attention_mask,
                "use_cache": use_cache}

    @staticmethod
    def _mask(attention_mask, offset, T, device):
        Tk = offset + T
        pad = attention_mask is not None and not bool(attention_mask[:, :Tk].all())
        if not pad and (offset == 0 or T == 1):
            return None
        rows = offset + torch.arange(T, device=device)[:, None]
        cols = torch.arange(Tk, device=device)[None, :]
        mask = (cols <= rows)[None, None]
        if pad:
            mask = (mask & attention_mask[:, None, None, :Tk].bool()) | (cols == rows)[None, None]
        return mask

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=None, labels=None, **kwargs):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        offset = past_key_values.get_seq_length() if past_key_values is not None else 0
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        cache = past_key_values if use_cache else None
        ids = input_ids[:, offset:]
        B, T = ids.shape
        mask = self._mask(attention_mask, offset, T, ids.device)
        x0 = _norm(self.tok_emb(ids).bfloat16())
        tables = self.value_emb(ids).bfloat16().chunk(self.config.num_value_tables, dim=-1)
        x, saved = x0, {}
        for i, layer in enumerate(self.layers):
            if i in self._skip_into:
                n, src = self._skip_into[i]
                x = x + self.skip_scale[n] * saved[src]
            ve = tables[self._ve_table[i]] if i in self._ve_table else None
            x = layer(x, x0, ve, mask, cache, offset)
            if i in self._skip_from:
                saved[i] = x
        logits = self.head(_norm(x)).float()
        cap = self.config.final_logit_softcapping
        if cap is not None and cap > 0:
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, offset + 1:].reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=cache)
