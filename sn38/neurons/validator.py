"""
SN38 Validator — Two-stage evaluation

Stage 1: Leak detection (all miners)
Stage 2: Quality evaluation via elimination bracket (top N only)

Winner takes all.

Usage:
    python -m sn38.neurons.validator --netuid 38
"""

import argparse
import hashlib
import logging
import os
import time
import tempfile

import numpy as np
import torch
import bittensor as bt

logger = logging.getLogger(__name__)

from ..template.model_loader import load_model
from ..template.constants import NETWORKS
from ..template.model_store import download_model, parse_repo, get_repo_file_size, count_model_params, get_device, verify_commit_sha
from ..template.backend_api import BackendAPI
from ..template.validator_db import get_connection, get_cached_result, save_result, is_week_evaluated, mark_week_evaluated, cleanup_after_uid, get_unsynced_eval_details, mark_synced
from ..template.cosine_gate import check_cosine_gate, load_baselines, unload_baselines
from ..template.leak import evaluate
from ..template.quality import run_quality_duels
from ..template.round_results import RoundResults

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def _free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _sync_eval_details(api, conn):
    unsynced = get_unsynced_eval_details(conn)
    if not unsynced:
        return
    logger.info(f"Syncing {len(unsynced)} eval details to backend...")
    synced = 0
    for row in unsynced:
        if api.submit_eval_detail(row["round"], row["uid"], row["year"], row["repo_id"],
                                  row["passed"], row["score"], row["score_unknown"], row["score_known"]):
            mark_synced(conn, row["uid"], row["year"], row["repo_id"], row["round"])
            synced += 1
    logger.info(f"Sync complete: {synced}/{len(unsynced)}")


def _preload_benchmarks(api, all_years):
    benchmarks = {}
    for year in all_years:
        benchmarks[year] = {
            "unknown": api.get_benchmark(year),
            "known": api.get_benchmark(year, known=True),
        }
    return benchmarks


def check_duplicate_weights(api, model_path, uid, snapshot_at):
    """Check if model weights were already submitted by another miner. Returns True if allowed."""
    model_file = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(model_file):
        model_file = os.path.join(model_path, "pytorch_model.bin")
    weight_hash = hashlib.sha256(open(model_file, "rb").read()).hexdigest()
    check = api.check_hash(weight_hash, uid, snapshot_at)
    if not check["allowed"]:
        logger.warning(f"UID {uid}: duplicate weights (owner: UID {check['owner_uid']}), skipping")
        return False
    return True


def run_stage1(api, submissions, submission_times, config, all_years, conn, benchmarks, eval_round):
    """Evaluate all miners for leak detection. Returns {uid: score}."""
    WORST_SCORE = 0.0
    leak_scores = {}
    owner_uid = config.get("owner_uid", 0)

    device = get_device()
    logger.info("Loading baselines for cosine gate...")
    load_baselines(all_years, device)

    sorted_uids = sorted(submissions.keys(), key=lambda u: submission_times.get(u, "9999"))
    total = len(sorted_uids)
    for i, uid in enumerate(sorted_uids):
        models = submissions[uid]
        logger.info(f"UID {uid}: {len(models)} years submitted")

        repo_to_years = {}
        year_scores = {year: WORST_SCORE for year in all_years}

        cached_count = 0
        for year in all_years:
            repo_id = models.get(str(year))
            if not repo_id:
                continue
            cached = get_cached_result(conn, uid, year, repo_id)
            if cached is not None:
                _, score = cached
                year_scores[year] = score
                cached_count += 1
                continue
            repo_to_years.setdefault(repo_id, []).append(year)

        if cached_count > 0:
            logger.info(f"UID {uid}: {cached_count} scores loaded from cache")

        for repo_str, years in repo_to_years.items():
            repo_id, revision = parse_repo(repo_str)
            file_size = get_repo_file_size(repo_id, revision)
            if file_size > config["max_model_bytes"]:
                logger.warning(f"UID {uid}: {repo_str} too large, skipping")
                continue

            def fail_year(y):
                save_result(conn, uid, y, repo_str, False, WORST_SCORE, 0.0, 0.0, eval_round)
                if api.submit_eval_detail(eval_round, uid, y, repo_str, False, WORST_SCORE, 0.0, 0.0):
                    mark_synced(conn, uid, y, repo_str, eval_round)

            def fail_repo_years():
                for y in years:
                    fail_year(y)

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    logger.info(f"UID {uid}: downloading {repo_id}...")
                    path = download_model(repo_id, tmpdir, revision=revision)

                    if not verify_commit_sha(repo_id, revision):
                        logger.warning(f"UID {uid}: revision {revision} is not a real commit SHA, skipping")
                        fail_repo_years()
                        continue

                    if not check_duplicate_weights(api, path, uid, submission_times.get(uid, "")):
                        fail_repo_years()
                        continue

                    eval_start = time.time()
                    model, tokenizer = load_model(path, device)
                    param_count = count_model_params(model)
                    logger.info(f"UID {uid}: loaded {param_count / 1e6:.0f}M params")

                    if param_count > config["max_parameters"]:
                        logger.warning(f"UID {uid}: {param_count / 1e9:.1f}B > limit, skipping")
                        del model
                        _free_gpu()
                        fail_repo_years()
                        continue

                    cosine_results = {}
                    if uid != owner_uid:
                        candidate_state = model.state_dict()
                        for year in years:
                            gate_passed, avg_cosine = check_cosine_gate(candidate_state, year)
                            cosine_results[year] = gate_passed
                            logger.info(f"UID {uid}: year {year} cosine={avg_cosine:.6f} gate={'PASS' if gate_passed else 'FAIL'}")
                        del candidate_state

                    for year in years:
                        if time.time() - eval_start > config["max_eval_seconds"]:
                            logger.warning(f"UID {uid}: timeout, remaining years skipped")
                            break

                        if not cosine_results.get(year, True):
                            fail_year(year)
                            continue

                        bench = benchmarks[year]
                        logger.info(f"UID {uid}: evaluating year {year}...")
                        failed_leak, median_unknown = evaluate(model, device, bench["unknown"], tokenizer)
                        passed_known, median_known = evaluate(model, device, bench["known"], tokenizer)
                        passed = not failed_leak and passed_known

                        if not passed:
                            score = WORST_SCORE
                        else:
                            score = median_unknown - median_known
                        year_scores[year] = score
                        save_result(conn, uid, year, repo_str, passed, score, median_unknown, median_known, eval_round)
                        if api.submit_eval_detail(eval_round, uid, year, repo_str, passed, score, median_unknown, median_known):
                            mark_synced(conn, uid, year, repo_str, eval_round)
                        logger.info(f"UID {uid}: year {year} {'PASSED' if passed else 'FAILED'}")
                        logger.debug(f"UID {uid} year {year}: unknown={median_unknown:.4f} known={median_known:.4f} score={score:.4f}")

                    elapsed = time.time() - eval_start
                    logger.info(f"UID {uid}: done in {elapsed:.0f}s")
                    del model
                    _free_gpu()

            except RuntimeError:
                raise
            except Exception as e:
                logger.error(f"UID {uid}: {repo_id} FAILED — {type(e).__name__}")
                fail_repo_years()

        leak_scores[uid] = sum(year_scores.values()) / len(all_years)
        logger.debug(f"UID {uid}: leak_score={leak_scores[uid]:.4f}")
        logger.info(f"Stage 1 progress: {i + 1}/{total} ({(i + 1) / total:.0%})")

    unload_baselines()
    _sync_eval_details(api, conn)
    return leak_scores


def qualify(leak_scores, config):
    """Filter and normalize miners for Stage 2. Returns (qualified, normalized_leak)."""
    top_n = config.get("top_n_for_quality", 10)
    min_eval_score = config.get("min_eval_score", -3.0)
    ranked = sorted(leak_scores.items(), key=lambda x: x[1])
    qualified = [(uid, score) for uid, score in ranked if score < min_eval_score][:top_n]

    logger.info(f"Qualified: {len(qualified)} miners — UIDs: {[uid for uid, _ in qualified]}")

    eval_threshold = config.get("min_eval_score", -3.0)
    eval_best = config.get("leak_epsilon", -6.0)
    normalized_leak = {uid: max(0.0, min(1.0, (eval_threshold - score) / (eval_threshold - eval_best))) for uid, score in qualified}

    return qualified, normalized_leak


def run_stage2_and_score(api, leak_scores, submissions, submission_times, config, all_years, metagraph):
    """Run qualification, quality duels, and compute final scores."""
    qualified, normalized_leak = qualify(leak_scores, config)

    if not qualified:
        owner_uid = config.get("owner_uid", 0)
        results = RoundResults.no_qualified(leak_scores, owner_uid)
        return np.zeros(metagraph.n), None, [owner_uid], [1.0], results

    win_rates = None
    if len(qualified) == 1:
        logger.info("Only 1 miner qualified, skipping stage 2")
        final_scores = np.zeros(metagraph.n)
        final_scores[qualified[0][0]] = 1.0
    else:
        logger.info("=== Stage 2: Quality evaluation ===")
        questions = api.get_quality_questions()
        if not questions:
            logger.warning("No quality questions, skipping stage 2")
            final_scores = np.zeros(metagraph.n)
            for uid, score in qualified:
                final_scores[uid] = normalized_leak[uid]
        else:
            win_rates = run_quality_duels(qualified, submissions, questions, metagraph, all_years)
            leak_weight = config.get("leak_weight", 0.7)
            quality_weight = config.get("quality_weight", 0.3)
            final_scores = np.zeros(metagraph.n)
            for uid, _ in qualified:
                final_scores[uid] = leak_weight * normalized_leak[uid] + quality_weight * win_rates[uid]
                logger.info(f"UID {uid}: final={final_scores[uid]:.4f} (leak={normalized_leak[uid]:.4f} quality={win_rates[uid]:.4f})")

    winner = None
    if final_scores.sum() > 0:
        max_score = final_scores.max()
        tied_uids = [uid for uid in range(metagraph.n) if final_scores[uid] == max_score]
        if len(tied_uids) > 1:
            winner = min(tied_uids, key=lambda u: submission_times.get(u, "9999"))
            logger.info(f"Tie between UIDs {tied_uids}, earliest submission wins")
        else:
            winner = tied_uids[0]
        logger.info(f"Winner: UID {winner} score={final_scores[winner]:.4f}")

    # Emission split: winner gets emission_pct, owner gets the rest
    emission_pct = config.get("emission_pct", 0.30)
    owner_uid = config.get("owner_uid", 0)
    uids = []
    weights = []

    if winner is not None and winner != owner_uid:
        uids.append(winner)
        weights.append(emission_pct)
        if emission_pct < 1.0:
            uids.append(owner_uid)
            weights.append(1.0 - emission_pct)
    else:
        uids.append(owner_uid)
        weights.append(1.0)

    results = RoundResults.with_winner(leak_scores, qualified, win_rates, final_scores, winner, uids, weights)

    return final_scores, winner, uids, weights, results


def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(message)s",
    )

    api = BackendAPI(BACKEND_URL)

    config = api.get_config()
    ALL_YEARS = api.get_years()
    NUM_YEARS = len(ALL_YEARS)
    eval_round = api.get_eval_round()
    logger.info(f"Config: {config}")
    logger.info(f"Eval round: {eval_round}")

    netuid = NETWORKS[args.network]["netuid"]
    owner_uid = NETWORKS[args.network]["owner_uid"]

    conn = get_connection()
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(netuid=netuid)

    _sync_eval_details(api, conn)

    if is_week_evaluated(conn, eval_round):
        logger.info(f"Round {eval_round} already evaluated, skipping")
        return

    submissions, submission_times = api.get_submissions(eval_round)
    if not submissions:
        logger.info(f"Round {eval_round}: no submissions")
        mark_week_evaluated(conn, eval_round)
        return

    if args.test_uids:
        test_uids = set(int(u) for u in args.test_uids.split(","))
        submissions = {uid: m for uid, m in submissions.items() if uid in test_uids}

    logger.info(f"Round {eval_round}: {len(submissions)} miners")

    # =========================================
    # Preload benchmarks (fail fast if backend is down)
    # =========================================
    benchmarks = _preload_benchmarks(api, ALL_YEARS)
    logger.info(f"Loaded benchmarks for {len(benchmarks)} years")

    # =========================================
    # STAGE 1: Leak detection
    # =========================================
    logger.info("=== Stage 1: Leak detection ===")
    leak_scores = run_stage1(api, submissions, submission_times, config, ALL_YEARS, conn, benchmarks, eval_round)

    # =========================================
    # STAGE 2: Quality evaluation (round-robin)
    # =========================================
    config["owner_uid"] = owner_uid
    final_scores, winner, uids, weights, results = run_stage2_and_score(
        api, leak_scores, submissions, submission_times, config, ALL_YEARS, metagraph
    )

    api.submit_eval_results(eval_round, results.to_dict())

    subtensor.set_weights(
        wallet=wallet, netuid=netuid,
        uids=uids, weights=weights,
        wait_for_inclusion=False,
    )
    logger.info(f"Weights set: {dict(zip(uids, weights))}")

    mark_week_evaluated(conn, eval_round)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wallet.name", type=str, default="validator", dest="wallet_name")
    parser.add_argument("--wallet.hotkey", type=str, default="default", dest="wallet_hotkey")
    parser.add_argument("--subtensor.network", type=str, default="finney", dest="network")
    parser.add_argument("--test-uids", type=str, default=None, dest="test_uids",
                        help="Comma-separated UIDs to evaluate (e.g. --test-uids 2,3,5)")
    args = parser.parse_args()
    run(args)
