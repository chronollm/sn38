from transformers import PretrainedConfig


class ChronoGrowConfig(PretrainedConfig):
    model_type = "sn38-chronogrow"

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=1536,
        num_hidden_layers=52,
        num_attention_heads=12,
        max_position_embeddings=1792,
        final_logit_softcapping=15.0,
        num_value_tables=3,
        skips=None,
        value_embed_layers=None,
        growth=None,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=50256,
        tie_word_embeddings=False,
        **kwargs,
    ):
        L = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = L
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.final_logit_softcapping = final_logit_softcapping
        self.num_value_tables = num_value_tables
        if skips is None:
            skips = [[L - 1 - m, m] for m in range(L // 2, L)]
        if value_embed_layers is None:
            value_embed_layers = [[i, i] for i in range(3)] + [[L - 3 + i, i] for i in range(3)]
        self.skips = [[int(s), int(d)] for s, d in skips]
        self.value_embed_layers = [[int(l), int(t)] for l, t in value_embed_layers]
        dsts = [d for _, d in self.skips]
        if len(set(dsts)) != len(dsts) or any(not 0 <= s < d < L for s, d in self.skips):
            raise ValueError(f"skips need unique targets and source < target < {L}, got {self.skips}")
        if any(not (0 <= l < L and 0 <= t < num_value_tables) for l, t in self.value_embed_layers):
            raise ValueError(f"bad value_embed_layers: {self.value_embed_layers}")
        self.growth = [self._entry(g, L) for g in (growth or [])]
        years = [g["year"] for g in self.growth]
        if years != sorted(set(years)):
            raise ValueError(f"growth years must be unique and ascending, got {years}")
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @staticmethod
    def _entry(g, num_layers):
        layers = g.get("layers")
        layers = list(range(num_layers)) if layers is None else sorted(set(int(i) for i in layers))
        if int(g["width"]) <= 0 or not layers or layers[0] < 0 or layers[-1] >= num_layers:
            raise ValueError(f"bad growth entry: {g}")
        return {"year": int(g["year"]), "width": int(g["width"]), "layers": layers}

    def growth_for_layer(self, i):
        return [(g["year"], g["width"]) for g in self.growth if i in g["layers"]]
