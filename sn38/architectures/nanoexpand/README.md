# NanoExpand

`model_type: sn38-nanoexpand`

Nanochrono with optional extra MLP blocks at selected depths. With no blocks configured it is
identical to nanochrono; the addition is opt-in through a single config field.

## Motivation

Each round targets a later cutoff year, and the known set spans many years — the cutoff year is
the majority but earlier years are a large minority. A model therefore has to acquire the new year
*and* retain the previous ones. Losing the earlier years fails the known gate just as surely as
never learning the new one.

With a fixed architecture those two goals compete for the same parameters: the new year is written
into weights that already hold the earlier ones. The extra blocks give each new cutoff year
dedicated capacity to learn into, so acquisition and retention are not fighting over the same
weights. They can also be frozen in later training stages to hold what they have learned while the
rest of the model continues to train.

## What changes

One config field:

```json
{
  "model_type": "sn38-nanoexpand",
  "extra_mlp_after": [12, 14, 16]
}
```

`extra_mlp_after` lists decoder-layer indices. After each listed layer, one additional residual
MLP block runs:

```python
x = layer(x, aux, cos_sin, window, mask, cache, past_len)
if str(i) in self.extra_mlps:
    x = x + self.extra_mlps[str(i)](_norm(x))
```

Each block has the same form as the existing MLPs — `hidden → intermediate → hidden`,
squared-ReLU, no gate:

```python
down_proj(F.relu(up_proj(x)).square())
```

Everything else — attention, auxiliary value embeddings, sliding-window pattern, RoPE, QK
normalisation, residual and skip scales, the mid-stack tap, logit softcapping — is unchanged from
nanochrono.

## Parameter cost

`2 × hidden_size × intermediate_size` per block. At nanochrono's default dimensions
(1792 / 7168) that is **25,690,112** parameters each:

| blocks | added |
|---:|---:|
| 2 | 51.4M |
| 3 | 77.1M |
| 4 | 102.8M |

## Initialisation

Each block's `down_proj` is zero-initialised, so a newly extended model produces output identical
to the same model without the blocks. The extension adds capacity without perturbing the network,
so training starts from a stable state rather than a disrupted one.

Note that `up_proj` receives no gradient at step 0 — its gradient path runs through `down_proj`,
which is zero. The block therefore activates in two stages: `down_proj` moves first, then
`up_proj` follows. A higher learning rate on the extra blocks shortens that phase.

## Checkpoint compatibility

Decoder layers keep their original indices, and `residual_scales` / `skip_scales` keep their
original length. A nanochrono checkpoint therefore loads with `strict=False`; only the new
`extra_mlps` tensors are reported missing, and they retain their zero initialisation.

```python
model.load_state_dict(nanochrono_state_dict, strict=False)
# missing: model.extra_mlps.<i>.{up,down}_proj.weight
# unexpected: []
```

## Placement

`extra_mlp_after` accepts any layer indices in `[0, num_hidden_layers)`. Placing a block on the
layer named by `tap_layer` means the tap snapshot includes that block's output, which changes what
the final subtraction removes — avoid that index unless the effect is intended.

An empty or omitted `extra_mlp_after` yields exactly nanochrono.

## Files

| file | contents |
|---|---|
| `configuration_nanoexpand.py` | `NanoExpandConfig`, `model_type = "sn38-nanoexpand"` |
| `modeling_nanoexpand.py` | `NanoExpandModel`, `NanoExpandForCausalLM` |
| `__init__.py` | registers with `AutoConfig` / `AutoModelForCausalLM` |

Weights are safetensors, the tokenizer is standard `tokenizer.json`, and no code from a model repo
is executed — the architecture loads with `trust_remote_code=False`.
