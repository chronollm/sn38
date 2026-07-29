from transformers import PretrainedConfig


class NanochronoConfig(PretrainedConfig):
    model_type = "sn38-nanochrono"

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=1792,
        intermediate_size=7168,
        num_hidden_layers=28,
        num_attention_heads=14,
        num_key_value_heads=14,
        max_position_embeddings=2048,
        rope_theta=100000.0,
        sliding_window=512,
        layer_types=None,
        attention_multiplier=1.2,
        final_logit_softcapping=15.0,
        aux_layers=None,
        aux_gate_dim=12,
        mix_gate_dim=24,
        tap_layer=14,
        use_cache=True,
        bos_token_id=32759,
        eos_token_id=32763,
        pad_token_id=32763,
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
        self.sliding_window = sliding_window
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        self.layer_types = list(layer_types)
        self.attention_multiplier = attention_multiplier
        self.final_logit_softcapping = final_logit_softcapping
        if aux_layers is None:
            aux_layers = [i for i in range(num_hidden_layers) if i % 2 == (num_hidden_layers - 1) % 2]
        self.aux_layers = list(aux_layers)
        self.aux_gate_dim = aux_gate_dim
        self.mix_gate_dim = mix_gate_dim
        self.tap_layer = tap_layer
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
