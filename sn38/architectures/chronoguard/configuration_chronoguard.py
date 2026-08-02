from transformers import PretrainedConfig


class ChronoguardConfig(PretrainedConfig):
    model_type = "sn38-chronoguard"

    def __init__(
        self,
        vocab_size=50304,
        num_layers=52,
        num_heads=12,
        model_dim=None,
        hidden_size=None,
        head_d_h=1024,
        head_n_blocks=10,
        head_top_k=8,
        head_low_rank_out=640,
        logit_cap=15.0,
        bos_token_id=None,
        eos_token_id=50256,
        pad_token_id=50256,
        use_cache=False,
        **kwargs,
    ):
        dim = model_dim if model_dim is not None else hidden_size
        if dim is None:
            dim = 1536
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.hidden_size = dim
        self.num_hidden_layers = num_layers
        self.head_d_h = head_d_h
        self.head_n_blocks = head_n_blocks
        self.head_top_k = head_top_k
        self.head_low_rank_out = head_low_rank_out
        self.logit_cap = logit_cap
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    @property
    def model_dim(self) -> int:
        return self.hidden_size

    def to_dict(self):
        output = super().to_dict()
        output.pop("model_dim", None)
        return output
