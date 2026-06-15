# Mining on SN38

## Overview

Train chronologically consistent ChronoGPT models and compete for emissions.

You train one model per year (2013-2025), upload them to HuggingFace, and submit a mapping on-chain. Validators evaluate your models for consistency and quality.

## Requirements

- Bittensor wallet registered on SN38
- HuggingFace account with a write token
- GPU for training (the models use the ChronoGPT architecture)

## Step 1: Train your models

Each model must use the **ChronoGPT architecture** and be trained only on data available up to its cutoff year. A 2018 model must not contain any knowledge from 2019 or later.

Each HuggingFace repo must contain:

```
config.json           # {"vocab_size": 50304, "num_layers": 52, "num_heads": 12, "model_dim": 1536}
model.safetensors     # weights in safetensors format
```

The validator loads models using its own trusted copy of the architecture. No code from your repo is executed.

## Step 2: Create your models.json

Map each year to a HuggingFace repo:

```json
{
  "2013": "your-username/chronogpt-2013",
  "2014": "your-username/chronogpt-2014",
  "2015": "your-username/chronogpt-2015"
}
```

To specify a branch, use `@`:

```json
{
  "2013": "your-username/chronogpt-2013@safetensors"
}
```

No `@` defaults to the `main` branch. You should submit all 13 years — missing years receive the worst possible score.

## Step 3: Register and submit

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo and install dependencies
git clone git@github.com:chronollm/sn38.git
cd sn38
uv sync

# Register on SN38
btcli subnet register --netuid 38 --wallet.name miner --wallet.hotkey default
```

### Option A: Auto-upload (recommended)

The script uploads your `models.json` to a HuggingFace dataset and commits the URL on-chain:

```bash
python -m sn38.neurons.miner \
  --wallet.name miner \
  --wallet.hotkey default \
  --models models.json \
  --hf-token hf_xxx
```

You can set `HF_TOKEN` as an environment variable instead of `--hf-token`.

### Option B: Existing dataset

If you already have a HuggingFace dataset repo with a `models.json`:

```bash
python -m sn38.neurons.miner \
  --wallet.name miner \
  --wallet.hotkey default \
  --dataset-repo your-username/sn38-submission
```

## Updating your models

Resubmit at any time with the same command. The backend polls the chain periodically and picks up the new submission. Model revisions are pinned at poll time — changing your model after the backend has read it won't affect the current round.

> **Tip**: You can keep your HuggingFace repos private during the round and make them public (or submit) just a few minutes before the round ends (Monday 12:00 UTC). This prevents other miners from copying your weights during the submission phase.

## Verify your submission

After submitting, the backend polls the chain every 5 minutes. You can verify your submission was picked up:

```bash
# Check current round
curl https://api.chronollm.com/rounds/current

# Check your submission (replace {round} and {uid} with your values)
curl https://api.chronollm.com/submissions/{round}/{uid}
```

## Scoring

Your models are evaluated in two stages:

1. **Consistency check** — each year is validated against a private dataset. The score reflects how well your model respects its temporal boundary. Missing years or errors receive the worst score.

2. **Quality evaluation** — the top 10 miners compete in round-robin duels judged by an LLM. The win rate becomes your quality score.

Final score: `0.7 * consistency_score + 0.3 * quality_win_rate`. Winner takes all.
