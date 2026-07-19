"""
SN38 Miner — Submit ChronoGPT models

Miners commit a HuggingFace dataset repo URL on-chain containing a models.json.
The backend fetches the JSON from HuggingFace.

Option 1: Provide a models.json file + HF token (auto-uploads to HuggingFace)
    python -m sn38.neurons.miner \
        --wallet.name miner --wallet.hotkey default \
        --models models.json --hf-token hf_xxx

Option 2: Provide an existing HuggingFace dataset repo (already contains models.json)
    python -m sn38.neurons.miner \
        --wallet.name miner --wallet.hotkey default \
        --dataset-repo user/sn38-submission
"""

import argparse
import json
import logging
import os

import bittensor as bt

logger = logging.getLogger(__name__)
from ..template.constants import NETWORKS
from ..template.model_store import (
    validate_models_json,
    upload_models_json,
    fetch_models_json,
    commit_metadata,
)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="SN38 Miner — submit ChronoGPT models")
    parser.add_argument("--wallet.name", type=str, required=True, dest="wallet_name")
    parser.add_argument("--wallet.hotkey", type=str, required=True, dest="wallet_hotkey")
    parser.add_argument("--subtensor.network", type=str, default="finney", dest="network")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", type=str, help="Path to models.json file (requires --hf-token)")
    group.add_argument("--dataset-repo", type=str, dest="dataset_repo",
                       help="Existing HuggingFace dataset repo containing models.json (e.g. user/sn38-submission)")

    parser.add_argument("--hf-token", type=str, default=None, dest="hf_token",
                        help="HuggingFace write token (or set HF_TOKEN env var). Required with --models")
    args = parser.parse_args()

    if args.models:
        # Option 1: upload models.json to HuggingFace
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.error("--hf-token or HF_TOKEN env var required when using --models")
            return

        with open(args.models) as f:
            models = json.load(f)

        try:
            missing = validate_models_json(models)
        except ValueError as e:
            logger.error(f"Invalid models: {e}")
            return

        if missing:
            logger.warning(f"Missing years (will score 0): {missing}")

        logger.info(f"Models to submit ({len(models)} years):")
        for year, repo in sorted(models.items()):
            logger.info(f"  {year}: {repo}")

        from huggingface_hub import HfApi
        hf_user = HfApi(token=hf_token).whoami()["name"]
        dataset_repo = f"{hf_user}/sn38-submission"
        upload_models_json(models, dataset_repo, token=hf_token)
        logger.info(f"Uploaded models.json to {dataset_repo}")

    else:
        # Option 2: use existing dataset repo
        dataset_repo = args.dataset_repo
        models = fetch_models_json(dataset_repo)

        try:
            missing = validate_models_json(models)
        except ValueError as e:
            logger.error(f"Invalid models.json in {dataset_repo}: {e}")
            return

        if missing:
            logger.warning(f"Missing years (will score 0): {missing}")

        logger.info(f"Using existing dataset: {dataset_repo} ({len(models)} years)")

    # Connect to Bittensor and commit
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    netuid = NETWORKS[args.network]["netuid"]

    with bt.Subtensor(network=args.network) as subtensor:
        metagraph = subtensor.metagraph(netuid=netuid)
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            logger.error(f"Not registered. Run: btcli subnet register --netuid {netuid}")
            return

        uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        commit_metadata(subtensor=subtensor, wallet=wallet, netuid=netuid, data=dataset_repo)
        logger.info(f"UID {uid}: committed {dataset_repo} on-chain. Done.")


if __name__ == "__main__":
    main()
