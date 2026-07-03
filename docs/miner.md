# Mining on SN38

## Overview

Train chronologically consistent ChronoGPT models and compete for emissions.

You train one model per year (2013-2024 included), upload them to HuggingFace, and submit a mapping on-chain. Validators evaluate your models for consistency and quality.

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

Map each year to a HuggingFace repo **pinned to a specific commit SHA**:

```json
{
  "2013": "your-username/chronogpt-2013@a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
  "2014": "your-username/chronogpt-2014@b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1",
  "2015": "your-username/chronogpt-2015@c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2"
}
```

Every repo must use the format `owner/repo@<40-char commit SHA>`. Branch names are not accepted. This ensures your model weights cannot be changed after submission.

To find your commit SHA, go to your repo on HuggingFace → Settings → History, or use:

```python
from huggingface_hub import HfApi
sha = HfApi().repo_info("your-username/chronogpt-2013").sha
print(sha)  # e.g. a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
```

You should submit all expected years — missing years receive the worst possible score.

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

Resubmit at any time with the same command. The backend polls the chain periodically and picks up the new submission.

> **Important**: Models must be pinned to a commit SHA. Once submitted, the weights behind that SHA are immutable — this prevents changes after the submission deadline.

## Private repos and timing

The HuggingFace **dataset repo** containing your `models.json` must always be **public** — the backend needs to read it at any time.

The individual **model repos** (the ones listed in `models.json`) can be kept **private** during the submission phase to prevent copying. They must be switched to **public within 1 hour after submissions close** so validators can download and evaluate them.

Timeline:
1. Keep your dataset repo (`models.json`) public at all times
2. Submit your models on-chain at any time (model repos can be private)
3. Submissions close Monday 12:00 UTC
4. Switch model repos to public before Monday 13:00 UTC
5. Validators download and evaluate

## Verify your submission

After submitting, the backend polls the chain every 5 minutes. You can verify your submission was picked up:

```bash
# Check current round
curl https://api.chronollm.com/rounds/current

# Check your submission (replace {round} and {uid} with your values)
curl https://api.chronollm.com/submissions/{round}/{uid}
```

## Anti-copy protection

Submitting someone else's model or an exact copy is pointless. During evaluation, the validator computes a hash of the model weights and checks that no other miner has submitted the same weights. The first submitter has priority — duplicates are rejected and receive the worst score. (Currently tied to miner UID, will be changed to hotkey in a future update.)

## Scoring

You can find all the variables of the scoring method at:[https://api.chronollm.com/docs](https://api.chronollm.com/docs)

Your models are evaluated in two stages:

1. **Consistency check** — each year is validated against a private dataset. The score reflects how well your model respects its temporal boundary. Missing years or errors receive the worst score.

2. **Quality evaluation** — the top 10 miners compete in round-robin duels judged by an LLM. The win rate becomes your quality score.

Final score: `0.7 * consistency_score + 0.3 * quality_win_rate`. Winner takes all.
