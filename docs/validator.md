# Validating on SN38

## Overview

Validators evaluate miner models and set weights on-chain. The validator runs inside a TEE (Trusted Execution Environment) on Phala Cloud to keep the evaluation dataset private.

## Hardware Requirements

For the first few weeks, we'll only be evaluating models up to 2B parameters, so CPU-only machines are sufficient since we're only running inference.

**Minimum:**
- 16 vCPUs
- 32 GB RAM
- 50 GB disk

**Recommended:**
- 32 vCPUs
- 64 GB RAM
- 50 GB disk

As the competition progresses, we'll ask miners to improve larger and larger models. At that point, we'll move to GPU-based evaluation.

Keep in mind that evaluation only runs once per week. Once it's finished, you're free to use the machine for other purposes or stop it to save costs.

## Prerequisites

- Registered validator on SN38 with sufficient stake
- [Phala Cloud](https://cloud.phala.com/) account with a TEE instance provisioned (CPU is enough for now, see above)
- OpenAI API key (for the LLM judge in Stage 2)
- HuggingFace token (for faster model downloads)

## Deploy

### 1. Install and log in to Phala CLI

```bash
npm install -g phala
phala login
```

### 2. Deploy the validator

```bash
phala deploy \
  -c docker-compose.validator.yml \
  --pre-launch-script scripts/prelaunch.sh \
  -t tdx.4xlarge \
  --image dstack-0.5.9 \
  -e HOTKEY_FILE_CONTENT="$(cat ~/.bittensor/wallets/validator/hotkeys/default)" \
  -e OPENAI_API_KEY=sk-xxx \
  -e HF_TOKEN=hf_xxx \
  --disk-size 150G \
  --no-dev-os
```

> **Tip**: Run `phala instance-types` to list all available instance types.

> **Note**: Replace the hotkey path with your actual wallet path (e.g. `~/.bittensor/wallets/<your-wallet>/hotkeys/<your-hotkey>`).

After deploying, Phala returns a CVM ID (e.g. `c797eb4a-86d6-4f27-a4d9-2973bd7a3d12`) and a URL to monitor the deployment from your browser. Save the CVM ID for later use.

### 3. Verify attestation

```bash
phala cvms attestation --cvm-id <your-cvm-id>
```

The `compose-hash` in the event log proves the validator is running the correct, unmodified code.

## Updating

To update the validator (e.g. after a new image is released), run the same deploy command with `--cvm-id`:

```bash
phala deploy \
  --cvm-id <your-cvm-id> \
  -c docker-compose.validator.yml \
  --pre-launch-script scripts/prelaunch.sh
```

## Monitoring

```bash
# View logs
phala cvms logs --cvm-id <your-cvm-id>

# Get CVM details
phala cvms get --cvm-id <your-cvm-id>
```

You can also monitor the validator from the Phala Cloud dashboard using the URL provided after deployment.

## Frequency

Rounds start every **Monday at 12:00 UTC**. Validators have **1 week** after the round starts to complete evaluation and set weights. The validator exits automatically after setting weights — no need to run it 24/7.

## Stage 1 parallelization

Stage 1 (leak detection) overlaps three things that used to run one miner at a time:

- the next miner's models download in the background while the current miner is validated/evaluated
- within a miner, models load into VRAM-sized batches and evaluate concurrently on separate CUDA streams
- duplicate-weight and SHA checks for a miner's repos run in parallel (CPU-only)

Loading rechecks live free VRAM before every model (not a running estimate — with same-sized models that drifts and can overcommit real memory) and reserves disk space the same way across every in-flight download, waiting if either is briefly unavailable rather than failing outright.

### Tuning env vars

| Variable | Default | Purpose |
|---|---|---|
| `TMP_DIR` | system temp dir | Where downloaded models land before eval. Point this at a disk with real headroom if `/tmp` is small or shared. |
| `MAX_CONCURRENT_EVALS` | `4` | Cap on models evaluated at the same instant within a batch. Loading only accounts for weight size, not the activation memory a forward pass needs — lower this if you see CUDA OOM errors during eval specifically (as opposed to during loading). |
| `DATA_DIR` | `/app/data` | Where the SQLite result cache lives. |
| `HF_HOME` / `HF_HUB_CACHE` | huggingface_hub default | Standard HF cache location for downloaded model blobs. |

### Local dry-run

`scripts/test_stage1_live.py` runs the real `run_stage1()` against real miners and real HuggingFace downloads, without needing a TEE deployment or touching subtensor/wallet/weight-setting — useful for testing changes to this pipeline before deploying. See the script's docstring for usage; it needs a local `leak_events.csv` since the real backend's `/benchmark` and `/models/check-hash` require TEE-attested mTLS that isn't available outside dstack.
