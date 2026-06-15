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
  -e OPENAI_API_KEY=sk-xxx
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
