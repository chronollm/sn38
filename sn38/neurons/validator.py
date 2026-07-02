"""
SN38 Validator — Two-stage evaluation

Stage 1: Leak detection (all miners)
Stage 2: Quality evaluation via elimination bracket (top N only)

Winner takes all.

Usage:
    python -m sn38.neurons.validator --netuid 38
"""

import argparse
import os
import time
import tempfile

import numpy as np
import bittensor as bt

from ..template.chronogpt_model import load_model
from ..template.constants import NETWORKS
from ..template.model_store import download_model, parse_repo, get_repo_file_size, count_model_params, get_device
from ..template.backend_api import BackendAPI
from ..template.validator_db import get_connection, get_cached_result, save_result, is_week_evaluated, mark_week_evaluated
from ..template.leak import evaluate
from ..template.quality import run_quality_duels

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def run(args):
    bt.logging.set_info()

    api = BackendAPI(BACKEND_URL)

    config = api.get_config()
    ALL_YEARS = api.get_years()
    NUM_YEARS = len(ALL_YEARS)
    eval_round = api.get_eval_round()
    bt.logging.info(f"Config: {config}")
    bt.logging.info(f"Eval round: {eval_round}")

    netuid = NETWORKS[args.network]["netuid"]
    owner_uid = NETWORKS[args.network]["owner_uid"]

    conn = get_connection()
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(netuid=netuid)

    if is_week_evaluated(conn, eval_round):
        bt.logging.info(f"Round {eval_round} already evaluated, skipping")
        return

    submissions, submission_times = api.get_submissions(eval_round)
    if not submissions:
        bt.logging.info(f"Round {eval_round}: no submissions")
        mark_week_evaluated(conn, eval_round)
        return

    if args.test_uids:
        test_uids = set(int(u) for u in args.test_uids.split(","))
        submissions = {uid: m for uid, m in submissions.items() if uid in test_uids}

    bt.logging.info(f"Round {eval_round}: {len(submissions)} miners")

    # =========================================
    # STAGE 1: Leak detection
    # =========================================
    bt.logging.info("=== Stage 1: Leak detection ===")

    WORST_SCORE = 0.0
    leak_scores = {}

    for uid, models in submissions.items():
        bt.logging.info(f"UID {uid}: {len(models)} years")

        repo_to_years = {}
        year_scores = {year: WORST_SCORE for year in ALL_YEARS}

        for year in ALL_YEARS:
            repo_id = models.get(str(year))
            if not repo_id:
                continue
            cached = get_cached_result(conn, uid, year, repo_id)
            if cached is not None:
                _, score = cached
                year_scores[year] = score
                continue
            repo_to_years.setdefault(repo_id, []).append(year)

        for repo_str, years in repo_to_years.items():
            repo_id, revision = parse_repo(repo_str)
            file_size = get_repo_file_size(repo_id, revision)
            if file_size > config["max_model_bytes"]:
                bt.logging.warning(f"UID {uid}: {repo_str} too large, skipping")
                continue

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = download_model(repo_id, tmpdir, revision=revision)

                    eval_start = time.time()
                    device = get_device()
                    model = load_model(path, device)
                    param_count = count_model_params(model)

                    if param_count > config["max_parameters"]:
                        bt.logging.warning(f"UID {uid}: {param_count / 1e9:.1f}B > limit, skipping")
                        del model
                        continue

                    for year in years:
                        if time.time() - eval_start > config["max_eval_seconds"]:
                            bt.logging.warning(f"UID {uid}: timeout, remaining years skipped")
                            break

                        benchmark_unknown = api.get_benchmark(year)
                        benchmark_known = api.get_benchmark(year, known=True)
                        if not benchmark_unknown:
                            bt.logging.warning(f"UID {uid}: no benchmark for year {year}, skipping")
                            continue

                        failed_leak, median_unknown = evaluate(model, device, benchmark_unknown)
                        passed_known, median_known = evaluate(model, device, benchmark_known)
                        passed = not failed_leak and passed_known

                        if not passed:
                            score = WORST_SCORE
                        else:
                            score = median_unknown - median_known
                        year_scores[year] = score
                        save_result(conn, uid, year, repo_id, passed, score)
                        bt.logging.info(f"UID {uid} year {year}: leak={not failed_leak} known={passed_known} unknown={median_unknown:.4f} known={median_known:.4f} score={score:.4f}")

                    del model

            except Exception as e:
                bt.logging.error(f"UID {uid}: {repo_id} FAILED — {type(e).__name__}")

        leak_scores[uid] = sum(year_scores.values()) / NUM_YEARS
        bt.logging.info(f"UID {uid}: leak_score={leak_scores[uid]:.4f}")

    # Qualify top N for Stage 2 (more negative = better, WORST_SCORE = 0.0)
    top_n = config.get("top_n_for_quality", 10)
    min_eval_score = config.get("min_eval_score", -20.0)
    ranked = sorted(leak_scores.items(), key=lambda x: x[1])  # most negative first
    qualified = [(uid, score) for uid, score in ranked if score < min_eval_score][:top_n]

    bt.logging.info(f"Stage 1 done: {len(qualified)} miners qualified")
    for uid, score in qualified:
        bt.logging.info(f"  UID {uid}: {score:.4f}")

    if not qualified:
        bt.logging.warning("No miners qualified — burning emissions")
        subtensor.set_weights(
            wallet=wallet, netuid=netuid,
            uids=[owner_uid], weights=[1.0],
            wait_for_inclusion=False,
        )
        mark_week_evaluated(conn, eval_round)
        return

    # Normalize leak scores to 0-1 with fixed bounds
    leak_min = config.get("min_eval_score", -20.0)  # best possible
    leak_max = config.get("leak_epsilon", -6.0)
    leak_range = leak_min - leak_max  # negative
    normalized_leak = {uid: max(0.0, min(1.0, (score - leak_max) / leak_range)) for uid, score in qualified}

    # =========================================
    # STAGE 2: Quality evaluation (round-robin)
    # =========================================
    if len(qualified) == 1:
        bt.logging.info("Only 1 miner qualified, skipping stage 2")
        final_scores = np.zeros(metagraph.n)
        final_scores[qualified[0][0]] = 1.0
    else:
        
        bt.logging.info("=== Stage 2: Quality evaluation ===")
        questions = api.get_quality_questions()
        if not questions:
            bt.logging.warning("No quality questions, skipping stage 2")
            final_scores = np.zeros(metagraph.n)
            for uid, score in qualified:
                final_scores[uid] = normalized_leak[uid]
        else:
            win_rates = run_quality_duels(qualified, submissions, questions, metagraph, ALL_YEARS)
            leak_weight = config.get("leak_weight", 0.7)
            quality_weight = config.get("quality_weight", 0.3)
            final_scores = np.zeros(metagraph.n)
            for uid, _ in qualified:
                final_scores[uid] = leak_weight * normalized_leak[uid] + quality_weight * win_rates[uid]
                bt.logging.info(f"UID {uid}: final={final_scores[uid]:.4f} (leak={normalized_leak[uid]:.4f} quality={win_rates[uid]:.4f})")

    # =========================================
    # Set weights — winner takes all
    # =========================================
    if final_scores.sum() > 0:
        max_score = final_scores.max()
        tied_uids = [uid for uid in range(metagraph.n) if final_scores[uid] == max_score]
        if len(tied_uids) > 1:
            winner = min(tied_uids, key=lambda u: submission_times.get(u, "9999"))
            bt.logging.info(f"Tie between UIDs {tied_uids}, earliest submission wins")
        else:
            winner = tied_uids[0]
        bt.logging.info(f"Winner: UID {winner} score={final_scores[winner]:.4f}")
        subtensor.set_weights(
            wallet=wallet, netuid=netuid,
            uids=[winner], weights=[1.0],
            wait_for_inclusion=False,
        )
        bt.logging.info(f"Weights set: UID {winner} = 1.0")
    else:
        bt.logging.warning("All scores are 0, no weights set")

    mark_week_evaluated(conn, eval_round)
    bt.logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet.name", type=str, default="validator", dest="wallet_name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", dest="wallet_hotkey")
    parser.add_argument("--subtensor.network", type=str, default="finney", dest="network")
    parser.add_argument("--test-uids", type=str, default=None, dest="test_uids",
                        help="Comma-separated UIDs to evaluate (e.g. --test-uids 2,3,5)")
    args = parser.parse_args()
    run(args)
