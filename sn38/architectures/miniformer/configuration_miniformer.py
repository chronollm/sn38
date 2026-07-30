from transformers import PretrainedConfig


class MiniformerConfig(PretrainedConfig):
    model_type = "sn38-miniformer"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
