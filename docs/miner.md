# Mining on SN38

## Overview

Train chronologically consistent language models and compete for emissions.

You train one model per year (2013-2024 included), upload them to HuggingFace, and submit a mapping on-chain. Validators evaluate your models for consistency and quality.

## Requirements

- Bittensor wallet registered on SN38
- HuggingFace account with a write token
- GPU for training

## Step 1: Train your models

Models can use **any architecture loadable by HuggingFace `AutoModelForCausalLM`** (Llama, Qwen, Gemma, Mistral, etc.). Each model must be trained only on data available up to its cutoff year. A 2018 model must not contain any knowledge from 2019 or later.

Each HuggingFace repo must contain:

```
config.json           # standard HuggingFace config
model.safetensors     # weights in safetensors format
tokenizer files       # tokenizer.json, tokenizer_config.json, etc.
```

The validator loads models with `trust_remote_code=False` — no code from your repo is executed.

### Test your architecture

Before submitting, verify your model is compatible with the evaluation pipeline:

```bash
python3 debug/test_automodel.py your-username/your-model
```

This runs both leak scoring and generation using the same code as the validator. If it works here, it will work on the subnet.

### Submit a custom architecture

If your model uses a custom architecture not built into HuggingFace `transformers`, you can request to have it added to the validator.

Since HuggingFace requires Python class definitions to load custom architectures, and the validator enforces `trust_remote_code=False`, custom architectures must be reviewed and added to the subnet codebase.

**What to submit** — open a PR adding a folder under `sn38/architectures/<your-architecture>/` with:

| File                      | Required | Description                                |
|---------------------------|----------|--------------------------------------------|
| `configuration_<name>.py` | Yes      | Config class inheriting `PretrainedConfig` |
| `modeling_<name>.py`      | Yes      | Model class inheriting `PreTrainedModel`   |

**Requirements:**

- Weights must be in `safetensors` format (no pickle)
- Tokenizer must use `tokenizer.json` (HF standard). No custom tokenizer classes or pickle files
- No arbitrary code execution in any file
- The `model_type` in `config.json` must be prefixed with `sn38-` (e.g. `sn38-mymodel`) to avoid conflicts with architectures built into HuggingFace

**Process:**

1. Open a PR with your architecture files, or send them via DM to the team
2. We review the code for security and correctness
3. Once added, any miner can use your architecture with `trust_remote_code=False`

**Why is this needed?** HuggingFace requires Python class definitions to instantiate custom architectures. The official way is to submit a PR to
the [transformers library](https://huggingface.co/docs/transformers/main/en/modular_transformers), but the review process can take weeks or months. Our architecture registry speeds this up — we review and add your architecture so you can
start competing immediately.

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

The individual **model repos** (the ones listed in `models.json`) can be kept **private** during the submission phase to prevent copying. They must be switched to **public within 1 hour after submissions close** so validators can download
and evaluate them.

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

## Self-test your model (leak detection)

Before submitting, you can test if your model passes the leak detection. The test runs inside a TEE using the exact same evaluation code as the validator, against a separate dataset from production. Returns only PASS/FAIL. Rate limited to
12 calls/day per hotkey. CPU only, no GPU needed.

Your hotkey must be registered on SN38. All environment variables (hotkey, HF token) are [encrypted locally](https://docs.phala.com/phala-cloud/cvm/set-secure-environment-variables) by the Phala CLI before upload and only decrypted inside
the TEE. No one, including Phala, can read them.

### 1. Install Phala CLI

```bash
npm install -g phala
phala login
```

### 2. Clone or update the subnet repo

The deploy command references files from the repo (`docker-compose.self-test.yml`, `scripts/prelaunch.sh`), so make sure you have the latest version:

```bash
git clone git@github.com:chronollm/sn38.git
cd sn38
git pull
```

### 3. Deploy and run the self-test

```bash
phala deploy \
  -c docker-compose.self-test.yml \
  --pre-launch-script scripts/prelaunch.sh \
  -e HOTKEY_FILE_CONTENT="$(cat ~/.bittensor/wallets/miner/hotkeys/default)" \
  -e HF_TOKEN=hf_xxx \
  -e HF_REPO=your-username/your-model@commit-sha \
  --instance-type tdx.xlarge \
  --no-dev-os
```

The test evaluates all years for the current submission round. A model that passes can expect to pass production. An overfit model will fail because the test uses a different dataset.

> **Instance type**: `tdx.xlarge` (8 vCPU, 16GB) is the recommended minimum. For faster inference, you can use a larger instance (e.g. `tdx.2xlarge`). List available CPU instances with `phala instance-types`.

## Anti-copy protection

Submitting someone else's model or an exact copy is pointless. During evaluation, the validator computes a hash of the model weights and checks that no other miner has submitted the same weights. Additionally, miners are compared pairwise using SVD spectral distance to detect near-duplicate models. The first submitter has priority, duplicates are rejected.

## Scoring

All evaluation parameters are available at [api.chronollm.com/config](https://api.chronollm.com/config).

Your models are evaluated in two main stages:

1. **Leak detection (pass/fail)**: each year is validated against a private dataset. The model must know its training era (pre-cutoff) and must not leak knowledge from after its cutoff year (post-cutoff). Missing years or errors receive the worst score.

2. **Quality duels**: qualified miners compete in round-robin 1v1 duels. Prompts are uniquely generated each round. Output quality is judged by an LLM. The win rate determines the final ranking.

The leak test is a gate. Among models that pass, quality determines the winner.
