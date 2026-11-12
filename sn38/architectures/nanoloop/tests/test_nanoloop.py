import json
import os

os.environ.setdefault("OPENAI_API_KEY", "test")

import pytest
import torch
from transformers import AutoModelForCausalLM

import sn38.architectures
from sn38.architectures.nanochrono.configuration_nanochrono import NanochronoConfig
from sn38.architectures.nanochrono.modeling_nanochrono import NanochronoForCausalLM
from sn38.architectures.nanoloop.configuration_nanoloop import NanoLoopConfig
from sn38.architectures.nanoloop.modeling_nanoloop import NanoLoopForCausalLM

SMALL = dict(
    vocab_size=128, hidden_size=64, intermediate_size=256, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    sliding_window=4, aux_layers=[1, 3, 5, 7], tap_layer=4,
    layer_types=["sliding_attention"] * 3 + ["full_attention"] * 1
                + ["sliding_attention"] * 3 + ["full_attention"] * 1,
    bos_token_id=120, eos_token_id=121, pad_token_id=121,
)
LOOP = dict(loop_start=4, loop_end=8, loop_count=2)
IDS = torch.tensor([[7, 21, 44, 3, 99, 12, 65, 8, 31, 77, 4, 55]])


def _randomize(model):
    torch.manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("skip_scales") or name.endswith("tap_scale") or name.endswith("mix_scale"):
                p.normal_(0, 0.1)


@pytest.fixture
def nano():
    torch.manual_seed(0)
    model = NanochronoForCausalLM(NanochronoConfig(**SMALL)).eval()
    _randomize(model)
    return model


def _loop(seed=0, **overrides):
    torch.manual_seed(seed)
    return NanoLoopForCausalLM(NanoLoopConfig(**SMALL, **{**LOOP, **overrides})).eval()


def _open(model):
    torch.manual_seed(2)
    with torch.no_grad():
        model.model.loop_gates.fill_(0.5)
        model.model.loop_embeds.normal_(0, 0.5)
    return model


def _logits(model, ids=IDS):
    with torch.no_grad():
        return model(input_ids=ids).logits


def _left_padded():
    short = IDS[:, 4:]
    pad = torch.full((1, 4), SMALL["pad_token_id"])
    batch = torch.cat([IDS, torch.cat([pad, short], 1)], 0)
    mask = torch.cat([torch.ones_like(IDS), torch.cat([torch.zeros_like(pad), torch.ones_like(short)], 1)], 0)
    return batch, mask


def test_no_loops_matches_nanochrono(nano):
    loop = _loop(loop_count=0)
    missing, unexpected = loop.load_state_dict(nano.state_dict(), strict=False)
    assert unexpected == []
    assert torch.equal(_logits(nano), _logits(loop))


def test_zero_gates_are_a_noop(nano):
    loop = _loop()
    loop.load_state_dict(nano.state_dict(), strict=False)
    assert torch.equal(_logits(nano), _logits(loop))
    assert torch.equal(
        nano.generate(IDS, max_new_tokens=8, do_sample=False, use_cache=True),
        loop.generate(IDS, max_new_tokens=8, do_sample=False, use_cache=True),
    )


def test_open_gates_change_output(nano):
    loop = _loop()
    loop.load_state_dict(nano.state_dict(), strict=False)
    _open(loop)
    assert not torch.equal(_logits(nano), _logits(loop))


def test_nanochrono_weights_load(nano):
    loop = _loop()
    missing, unexpected = loop.load_state_dict(nano.state_dict(), strict=False)
    assert unexpected == []
    assert sorted(missing) == ["model.loop_embeds", "model.loop_gates"]
    a, b = nano.state_dict(), loop.state_dict()
    for key in a:
        assert a[key].shape == b[key].shape


def test_param_cost():
    base = sum(p.numel() for p in _loop(loop_count=0).parameters())
    two = sum(p.numel() for p in _loop().parameters())
    assert two - base == 2 * (1 + SMALL["hidden_size"])


def test_fresh_model_has_zero_loop_tensors():
    loop = _loop()
    assert torch.count_nonzero(loop.model.loop_gates).item() == 0
    assert torch.count_nonzero(loop.model.loop_embeds).item() == 0


def test_from_pretrained_on_nanochrono_checkpoint(nano, tmp_path):
    nano.save_pretrained(tmp_path, safe_serialization=True)
    path = tmp_path / "config.json"
    cfg = json.loads(path.read_text())
    cfg.update(model_type="sn38-nanoloop", architectures=["NanoLoopForCausalLM"], **LOOP)
    path.write_text(json.dumps(cfg))
    for _ in range(3):
        junk = [torch.full((4096, 64), float("nan")) for _ in range(8)]
        del junk
        loop = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=False).eval()
        assert torch.count_nonzero(loop.model.loop_gates).item() == 0
        assert torch.count_nonzero(loop.model.loop_embeds).item() == 0
        assert torch.allclose(_logits(nano), _logits(loop), atol=1e-5)


def test_roundtrip_through_automodel(tmp_path):
    loop = _open(_loop())
    _randomize(loop)
    loop.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=False).eval()
    assert reloaded.config.loop_count == 2
    assert torch.allclose(_logits(loop), _logits(reloaded), atol=1e-5)


def test_incremental_forward_matches_full():
    loop = _open(_loop())
    _randomize(loop)
    with torch.no_grad():
        full = loop(input_ids=IDS, use_cache=False).logits
        cache = None
        for t in range(1, IDS.shape[1] + 1):
            out = loop(input_ids=IDS[:, :t], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            assert torch.allclose(full[:, t - 1], out.logits[:, -1], atol=1e-4)
    assert len(cache.layers) == SMALL["num_hidden_layers"] + 2 * 4


def test_causal():
    loop = _open(_loop())
    other = IDS.clone()
    other[0, -1] = 9
    a, b = _logits(loop), _logits(loop, other)
    assert torch.equal(a[:, :-1], b[:, :-1])
    assert not torch.equal(a[:, -1], b[:, -1])


def test_right_padding_does_not_change_scored_position():
    loop = _open(_loop())
    padded = torch.cat([IDS, torch.full((1, 5), SMALL["pad_token_id"])], 1)
    a, b = _logits(loop), _logits(loop, padded)
    assert torch.allclose(a[:, -1], b[:, IDS.shape[1] - 1], atol=1e-5)


def test_left_padded_batch_matches_nanochrono(nano):
    loop = _loop()
    loop.load_state_dict(nano.state_dict(), strict=False)
    batch, mask = _left_padded()
    with torch.no_grad():
        a = nano(input_ids=batch, attention_mask=mask).logits
        b = loop(input_ids=batch, attention_mask=mask).logits
    assert torch.equal(a, b)
    kwargs = dict(attention_mask=mask, max_new_tokens=6, do_sample=False, pad_token_id=SMALL["pad_token_id"])
    assert torch.equal(nano.generate(batch, **kwargs), loop.generate(batch, **kwargs))


def test_left_padded_batch_generate_runs_with_open_gates():
    loop = _open(_loop())
    _randomize(loop)
    batch, mask = _left_padded()
    out = loop.generate(batch, attention_mask=mask, max_new_tokens=6, do_sample=False,
                        pad_token_id=SMALL["pad_token_id"])
    assert out.shape == (2, IDS.shape[1] + 6)
    assert torch.equal(out[:, : IDS.shape[1]], batch)


def test_hidden_states_per_pass():
    loop = _open(_loop())
    with torch.no_grad():
        out = loop(input_ids=IDS, output_hidden_states=True)
    assert len(out.hidden_states) == 3
    assert not torch.equal(out.hidden_states[0], out.hidden_states[1])
    assert not torch.equal(out.hidden_states[1], out.hidden_states[2])
    with torch.no_grad():
        assert loop(input_ids=IDS).hidden_states is None


def test_gradients_at_init():
    loop = _loop()
    loop.train()
    loop.config.use_cache = False
    loop(input_ids=IDS, labels=IDS).loss.backward()
    assert loop.model.loop_gates.grad.abs().min().item() > 0.0
    assert loop.model.loop_embeds.grad.norm().item() == 0.0


@pytest.mark.parametrize("bad", [
    dict(loop_start=-1), dict(loop_end=9), dict(loop_start=6, loop_end=6),
    dict(loop_start=7, loop_end=5), dict(loop_count=-1),
])
def test_bad_loop_config_rejected(bad):
    with pytest.raises(ValueError):
        NanoLoopConfig(**SMALL, **{**LOOP, **bad})


def test_default_config_is_nanochrono():
    cfg = NanoLoopConfig(**SMALL)
    assert (cfg.loop_start, cfg.loop_end, cfg.loop_count) == (0, 8, 0)
