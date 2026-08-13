"""
Tests for the sn38-nanoexpand architecture.

    python -m pytest tests/test_nanoexpand.py -v
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test")

import pytest
import torch
from transformers import AutoModelForCausalLM

import sn38.architectures  # noqa: F401 — registers both architectures
from sn38.architectures.nanochrono.configuration_nanochrono import NanochronoConfig
from sn38.architectures.nanochrono.modeling_nanochrono import NanochronoForCausalLM
from sn38.architectures.nanoexpand.configuration_nanoexpand import NanoExpandConfig
from sn38.architectures.nanoexpand.modeling_nanoexpand import NanoExpandForCausalLM

SMALL = dict(
    vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    sliding_window=16, aux_layers=[1, 3, 5, 7], tap_layer=4,
    layer_types=["sliding_attention"] * 3 + ["full_attention"] * 1
                + ["sliding_attention"] * 3 + ["full_attention"] * 1,
    bos_token_id=120, eos_token_id=121, pad_token_id=121,
)
IDS = torch.tensor([[7, 21, 44, 3, 99, 12, 65, 8, 31, 77, 4, 55]])


@pytest.fixture
def nano():
    torch.manual_seed(0)
    return NanochronoForCausalLM(NanochronoConfig(**SMALL)).eval()


def _plus(extra, seed=0):
    torch.manual_seed(seed)
    return NanoExpandForCausalLM(NanoExpandConfig(**SMALL, extra_mlp_after=extra)).eval()


def _logits(model, ids=IDS):
    with torch.no_grad():
        return model(input_ids=ids).logits


# ─── Equivalence with nanochrono ───

def test_no_extras_matches_nanochrono(nano):
    """An empty extra_mlp_after is exactly nanochrono."""
    plus = _plus([])
    plus.load_state_dict(nano.state_dict(), strict=False)
    assert torch.equal(_logits(nano), _logits(plus))


def test_zero_init_is_a_noop(nano):
    """Blocks start zero-initialised, so output is unchanged until they are trained."""
    plus = _plus([3, 5])
    plus.load_state_dict(nano.state_dict(), strict=False)
    assert torch.equal(_logits(nano), _logits(plus))


def test_trained_blocks_change_output(nano):
    """Once down_proj is non-zero the blocks contribute."""
    plus = _plus([3, 5])
    plus.load_state_dict(nano.state_dict(), strict=False)
    with torch.no_grad():
        for block in plus.model.extra_mlps.values():
            block.down_proj.weight.normal_(0, 0.02)
    assert not torch.equal(_logits(nano), _logits(plus))


# ─── Checkpoint compatibility ───

def test_nanochrono_weights_load(nano):
    """Only the new tensors are missing; nothing is orphaned."""
    plus = _plus([3, 5])
    missing, unexpected = plus.load_state_dict(nano.state_dict(), strict=False)
    assert unexpected == []
    assert sorted(missing) == [
        "model.extra_mlps.3.down_proj.weight",
        "model.extra_mlps.3.up_proj.weight",
        "model.extra_mlps.5.down_proj.weight",
        "model.extra_mlps.5.up_proj.weight",
    ]


def test_decoder_indices_and_scale_shapes_unchanged(nano):
    """Extra blocks must not shift layer indices or resize the scalar vectors."""
    plus = _plus([3, 5])
    a, b = nano.state_dict(), plus.state_dict()
    for key in a:
        assert key in b, f"{key} disappeared"
        assert a[key].shape == b[key].shape, f"{key} changed shape"


def test_param_cost_per_block():
    """Each block costs 2 * hidden * intermediate."""
    base = sum(p.numel() for p in _plus([]).parameters())
    two = sum(p.numel() for p in _plus([3, 5]).parameters())
    assert two - base == 2 * (2 * SMALL["hidden_size"] * SMALL["intermediate_size"])


# ─── Validator load path ───

def test_roundtrip_through_automodel(tmp_path):
    """Saves and reloads with trust_remote_code=False, as the validator does."""
    plus = _plus([3, 5])
    with torch.no_grad():
        for block in plus.model.extra_mlps.values():
            block.down_proj.weight.normal_(0, 0.02)
    plus.save_pretrained(tmp_path, safe_serialization=True)

    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=False).eval()
    assert sorted(reloaded.model.extra_mlps.keys()) == ["3", "5"]
    assert torch.equal(_logits(plus), _logits(reloaded))


# ─── Generation (the Stage-2 duel path) ───

def test_generate_runs_with_cache():
    """generate() with a KV cache is the Stage-2 duel path.

    Note: do NOT assert cached == uncached on generated token IDs. On an
    untrained model the logits are near-uniform, so the ~1e-7 difference
    between the two code paths flips an argmax and the sequences diverge for
    reasons that have nothing to do with correctness. Cache correctness is
    covered numerically by test_incremental_forward_matches_full.
    """
    plus = _plus([3, 5])
    with torch.no_grad():
        for block in plus.model.extra_mlps.values():
            block.down_proj.weight.normal_(0, 0.02)
    out = plus.generate(IDS, max_new_tokens=8, do_sample=False, use_cache=True)
    assert out.shape == (1, IDS.shape[1] + 8)
    assert torch.equal(out[:, : IDS.shape[1]], IDS)


def test_matches_nanochrono_given_identical_weights():
    """With the same weights and zero-init blocks, behaviour is indistinguishable."""
    torch.manual_seed(0)
    nano = NanochronoForCausalLM(NanochronoConfig(**SMALL)).eval()
    plus = _plus([3, 5])
    plus.load_state_dict(nano.state_dict(), strict=False)
    assert torch.equal(_logits(nano), _logits(plus))
    assert torch.equal(
        nano.generate(IDS, max_new_tokens=8, do_sample=False, use_cache=True),
        plus.generate(IDS, max_new_tokens=8, do_sample=False, use_cache=True),
    )


def test_incremental_forward_matches_full():
    """A cached single-token step must agree with a full forward."""
    plus = _plus([3, 5])
    with torch.no_grad():
        for block in plus.model.extra_mlps.values():
            block.down_proj.weight.normal_(0, 0.02)
        full = plus(input_ids=IDS, use_cache=False).logits[:, -1]
        prefix = plus(input_ids=IDS[:, :-1], use_cache=True)
        step = plus(input_ids=IDS, past_key_values=prefix.past_key_values,
                    use_cache=True).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-4)


# ─── Gradient flow ───

def test_up_proj_has_no_gradient_at_init():
    """down_proj is zero, so up_proj's gradient path is zero until it moves."""
    plus = _plus([3, 5])
    plus.train()
    plus.config.use_cache = False
    plus(input_ids=IDS, labels=IDS).loss.backward()
    for i in ("3", "5"):
        block = plus.model.extra_mlps[i]
        assert block.up_proj.weight.grad.norm().item() == 0.0
        assert block.down_proj.weight.grad.norm().item() > 0.0


# ─── Config validation ───

@pytest.mark.parametrize("bad", [[-1], [8], [3, 99]])
def test_out_of_range_index_rejected(bad):
    with pytest.raises(ValueError):
        NanoExpandConfig(**SMALL, extra_mlp_after=bad)


def test_indices_are_deduped_and_sorted():
    cfg = NanoExpandConfig(**SMALL, extra_mlp_after=[5, 3, 5])
    assert cfg.extra_mlp_after == [3, 5]
