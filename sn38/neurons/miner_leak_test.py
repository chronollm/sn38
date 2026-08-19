"""Self-test — miners check if their model passes the leak test.

Runs inside TEE, same evaluation code as the validator.
Authenticates with the backend via TEE session.

Usage:
    python -m sn38.neurons.self_test \
        --wallet.name miner --wallet.hotkey default \
        --repo owner/model@commitsha
"""

import argparse
import logging
import os
import tempfile

import torch
import bittensor as bt

from ..template.backend_api import BackendAPI
from ..template.constants import NETWORKS
from ..template.model_loader import load_model
from ..template.leak import evaluate
from ..template.model_store import download_model, parse_repo, get_device

logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "https://api.chronollm.com")


def run(args):
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s | %(message)s")
    logging.getLogger("sn38").setLevel(logging.INFO)
    logging.getLogger(__name__).setLevel(logging.INFO)

    logger.info("[1/6] Loading wallet...")
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    hotkey = wallet.hotkey.ss58_address
    logger.info(f"[1/6] Hotkey: {hotkey}")

    logger.info("[2/6] Verifying hotkey on metagraph...")
    netuid = NETWORKS["finney"]["netuid"]
    for attempt in range(5):
        try:
            subtensor = bt.Subtensor(network="finney")
            metagraph = subtensor.metagraph(netuid=netuid)
            break
        except Exception as e:
            if attempt < 2:
                logger.warning(f"[2/6] Connection failed ({e}), retrying in 10s...")
                import time
                time.sleep(10)
            else:
                logger.error(f"[2/6] Failed to connect after 3 attempts: {e}")
                return
    if hotkey not in metagraph.hotkeys:
        logger.error(f"Hotkey {hotkey} is not registered on SN{netuid}")
        return
    uid = metagraph.hotkeys.index(hotkey)
    coldkey = metagraph.coldkeys[uid]
    logger.info(f"[2/6] Hotkey verified on SN{netuid} (uid={uid}, coldkey={coldkey[:12]}...)")

    logger.info("[3/6] Connecting to backend...")
    api = BackendAPI(BACKEND_URL, hotkey=hotkey)
    config = api.get_config()
    logger.info(f"[3/6] Config: {config}")

    try:
        api.check_leak_test_rate_limit(hotkey, args.repo, coldkey)
    except RuntimeError as e:
        logger.error(str(e))
        return

    logger.info(f"[3/6] Rate limit OK")

    current_round = api.get_submission_round()
    years = api.get_years(current_round)
    logger.info(f"[3/6] Submission round: {current_round} — years: {years}")
    device = get_device()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_id, revision = parse_repo(args.repo)
            logger.info(f"[4/6] Downloading {repo_id}...")
            path = download_model(repo_id, tmpdir, revision=revision)
            logger.info(f"[4/6] Download complete")

            logger.info("[5/6] Loading model...")
            model, _ = load_model(path, device)
            logger.info(f"[5/6] Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

            logger.info(f"[6/6] Evaluating {len(years)} year(s)...")
            all_passed = True
            year_results = []
            for i, year in enumerate(years, 1):
                logger.info(f"[6/6] Year {year} ({i}/{len(years)}) — fetching benchmark...")
                bench_unknown = api.get_selftest_benchmark(year)
                bench_known = api.get_selftest_benchmark(year, known=True)

                logger.info(f"[6/6] Year {year} ({i}/{len(years)}) — scoring...")
                failed_leak, median_unknown = evaluate(model, device, bench_unknown)
                passed_known, median_known = evaluate(model, device, bench_known)

                leak_ok = not failed_leak
                known_ok = passed_known
                passed = leak_ok and known_ok
                year_results.append((year, leak_ok, known_ok))
                logger.info(f"[6/6] Year {year} ({i}/{len(years)}) — {'PASS' if passed else 'FAIL'}")

                if not passed:
                    all_passed = False

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Evaluation failed: {type(e).__name__}: {e}")
        return

    print(f"\n{'=' * 40}")
    print(f"  Model: {args.repo}")
    print(f"  ----------------------------------------")
    for year, leak_ok, known_ok in year_results:
        print(f"  Year {year}:")
        print(f"    Leak (post-cutoff):  {'PASS' if leak_ok else 'FAIL'}")
        print(f"    Known (pre-cutoff):  {'PASS' if known_ok else 'FAIL'}")
    print(f"  ----------------------------------------")
    print(f"  Result: {'PASS' if all_passed else 'FAIL'}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet.name", type=str, default="miner", dest="wallet_name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", dest="wallet_hotkey")
    parser.add_argument("--repo", type=str, required=True, help="HF repo (owner/model@sha)")
    args = parser.parse_args()
    run(args)