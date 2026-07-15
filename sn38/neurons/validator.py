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
import os
import shutil
import threading
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

import numpy as np
import torch
import bittensor as bt

from ..template.chronogpt_model import load_model
from ..template.constants import NETWORKS
from ..template.model_store import download_model, parse_repo, get_repo_file_size, count_model_params, get_device, verify_commit_sha
from ..template.backend_api import BackendAPI
from ..template.validator_db import get_connection, get_cached_result, save_result, is_week_evaluated, mark_week_evaluated, get_unsynced_eval_details, mark_synced
from ..template.leak import evaluate
from ..template.quality import run_quality_duels
from ..template.round_results import RoundResults

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
TMP_DIR = os.environ.get("TMP_DIR")
MAX_CONCURRENT_EVALS = int(os.environ.get("MAX_CONCURRENT_EVALS", "4"))

# Multiple repos download concurrently per miner, and the next miner's
# downloads prefetch while the current miner's models are still on disk
# (through validation/load/eval) — this tracks bytes reserved for all of
# that so concurrent downloads can't collectively overshoot free disk space.
_disk_lock = threading.Lock()
_disk_reserved_bytes = 0


def _get_free_disk():
    return shutil.disk_usage(TMP_DIR or tempfile.gettempdir()).free


def _reserve_disk_space(repo_str, needed_bytes, poll_interval=5, max_wait_seconds=300):
    """Block until `needed_bytes` can be reserved without overshooting free
    disk space, accounting for every other download/on-disk model in flight.
    Caller must release the same amount once the directory is deleted."""
    global _disk_reserved_bytes
    waited = 0.0
    while True:
        with _disk_lock:
            if _get_free_disk() - _disk_reserved_bytes >= needed_bytes:
                _disk_reserved_bytes += needed_bytes
                return
        if waited >= max_wait_seconds:
            raise RuntimeError(f"{repo_str}: not enough disk space after {max_wait_seconds}s wait")
        bt.logging.info(f"{repo_str}: waiting for {needed_bytes / 1e9:.1f}GB free disk space...")
        time.sleep(poll_interval)
        waited += poll_interval


def _release_disk_space(reserved_bytes):
    global _disk_reserved_bytes
    with _disk_lock:
        _disk_reserved_bytes -= reserved_bytes


def _free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _sync_eval_details(api, conn):
    unsynced = get_unsynced_eval_details(conn)
    if not unsynced:
        return
    bt.logging.info(f"Syncing {len(unsynced)} eval details to backend...")
    synced = 0
    for row in unsynced:
        if api.submit_eval_detail(row["round"], row["uid"], row["year"], row["repo_id"],
                                  row["passed"], row["score"], row["score_unknown"], row["score_known"]):
            mark_synced(conn, row["uid"], row["year"], row["repo_id"], row["round"])
            synced += 1
    bt.logging.info(f"Sync complete: {synced}/{len(unsynced)}")


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
        bt.logging.warning(f"UID {uid}: duplicate weights (owner: UID {check['owner_uid']}), skipping")
        return False
    return True


def _download_all_for_uid(uid, models, all_years, config, conn):
    """Download every not-yet-cached repo for a UID in parallel. Returns (cached_scores, downloaded)."""
    repo_to_years = {}
    cached_scores = {}
    for year in all_years:
        repo_str = models.get(str(year))
        if not repo_str:
            continue
        cached = get_cached_result(conn, uid, year, repo_str)
        if cached is not None:
            cached_scores[year] = cached[1]
            continue
        repo_to_years.setdefault(repo_str, []).append(year)

    downloaded = {}
    if not repo_to_years:
        return cached_scores, downloaded

    def _download_one(repo_str, years):
        repo_id, revision = parse_repo(repo_str)
        file_size = get_repo_file_size(repo_id, revision)
        if file_size > config["max_model_bytes"]:
            raise ValueError("too_large")

        reserved = file_size * 1.3
        _reserve_disk_space(repo_str, reserved)
        tmpdir = None
        try:
            tmpdir = tempfile.mkdtemp(dir=TMP_DIR)
            download_model(repo_id, tmpdir, revision=revision)
            return repo_str, {"path": tmpdir, "years": years, "size": reserved}
        except Exception:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            _release_disk_space(reserved)
            raise

    with ThreadPoolExecutor(len(repo_to_years)) as pool:
        futures = {
            pool.submit(_download_one, rs, yrs): (rs, yrs)
            for rs, yrs in repo_to_years.items()
        }
        for f in as_completed(futures):
            rs, yrs = futures[f]
            try:
                _, result = f.result()
                downloaded[rs] = result
            except Exception as e:
                downloaded[rs] = {"error": e, "years": yrs}

    return cached_scores, downloaded


VRAM_RESERVE_CAP = 6 * 1024**3  # 6GB
VRAM_RESERVE_FRACTION = 0.2


def _get_free_vram():
    """VRAM usable for models, after reserving a safety margin — whichever is
    smaller of a flat 6GB or 20% of total device memory. A flat 20% wastes
    far more headroom than models actually need on a large card (e.g. 16GB
    on an 80GB GPU); the cap keeps small GPUs proportionally covered too."""
    if not torch.cuda.is_available():
        return float("inf")
    free, total = torch.cuda.mem_get_info()
    reserve = min(VRAM_RESERVE_CAP, total * VRAM_RESERVE_FRACTION)
    return max(0, free - reserve)


def _model_file_size(path):
    return sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith((".safetensors", ".bin"))
    )


def _load_batch(repos, device, config, poll_interval=5, max_wait_seconds=300):
    """Load models until VRAM runs out. Returns (loaded, remaining, oversized).

    Rechecks *live* free VRAM before every load instead of decrementing a
    locally-tracked estimate — with same-sized models loaded back to back,
    that estimate's error compounds across the batch and eventually
    overcommits real GPU memory (OOM), since it never reflects what's
    actually free at the moment of the next load.
    """
    loaded = []
    oversized = []

    for idx, (repo_str, path, years, size) in enumerate(repos):
        needed = _model_file_size(path) * 1.3
        waited = 0.0
        _free_gpu()
        while _get_free_vram() < needed:
            if loaded:
                # Won't fit alongside what's already in this batch — flush it
                # first so _free_batch() can release that VRAM, then retry.
                return loaded, repos[idx:], oversized
            if waited >= max_wait_seconds:
                bt.logging.warning(f"{repo_str}: no VRAM free after {max_wait_seconds}s, deferring to next batch")
                return loaded, repos[idx:], oversized
            bt.logging.info(f"{repo_str}: waiting for free VRAM...")
            time.sleep(poll_interval)
            waited += poll_interval
            # A prior batch's models may have been dereferenced but PyTorch's
            # caching allocator won't hand the memory back to the driver on
            # its own — without this, mem_get_info() keeps reporting it as
            # used and the wait never actually recovers on its own.
            _free_gpu()

        model = load_model(path, device)
        param_count = count_model_params(model)
        if param_count > config["max_parameters"]:
            bt.logging.warning(f"{repo_str}: {param_count / 1e9:.1f}B params > limit, skipping")
            del model
            _free_gpu()
            oversized.append((repo_str, years, path, size))
            continue

        loaded.append((repo_str, model, years, path, size))

    return loaded, [], oversized


def _free_batch(batch):
    for _, model, _, path, size in batch:
        del model
        shutil.rmtree(path, ignore_errors=True)
        _release_disk_space(size)
    _free_gpu()


def _eval_model(model, device, years, benchmarks):
    """Evaluate one model across its years on a dedicated CUDA stream."""
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    ctx = torch.cuda.stream(stream) if stream else nullcontext()
    results = {}
    with ctx:
        for year in years:
            bench = benchmarks.get(year)
            if not bench:
                continue
            failed_leak, median_unknown = evaluate(model, device, bench["unknown"])
            passed_known, median_known = evaluate(model, device, bench["known"])
            passed = not failed_leak and passed_known
            score = (median_unknown - median_known) if passed else 0.0
            results[year] = (passed, score, median_unknown, median_known, failed_leak, passed_known)
    if stream:
        stream.synchronize()
    return results


def _eval_batch(batch, device, benchmarks):
    """Evaluate all models in a batch in parallel using CUDA streams.

    Loading only checks VRAM for weight size, not the activation memory a
    forward pass needs — that's unpredictable up front (depends on sequence
    length etc.), so a batch that fits comfortably as weights can still OOM
    if every model in it runs its forward pass at the exact same instant.
    Capping concurrency here bounds how much activation memory piles up
    simultaneously, independent of how many models got loaded into the batch.
    """
    max_concurrent_evals = min(len(batch), MAX_CONCURRENT_EVALS)
    batch_years = {repo_str: years for repo_str, _, years, _, _ in batch}
    with ThreadPoolExecutor(max(1, max_concurrent_evals)) as pool:
        futures = {
            pool.submit(_eval_model, model, device, years, benchmarks): repo_str
            for repo_str, model, years, _, _ in batch
        }
        results = {}
        for f in as_completed(futures):
            repo_str = futures[f]
            try:
                results[repo_str] = f.result()
            except Exception as e:
                bt.logging.error(f"Eval error on {repo_str}: {e}")
                results[repo_str] = {y: (False, 0.0, 0.0, 0.0, None, None) for y in batch_years[repo_str]}
        return results


def _validate_repo(repo_str, path, uid, api, submission_times):
    """Pre-GPU checks: SHA verification + duplicate weights."""
    repo_id, revision = parse_repo(repo_str)
    if not verify_commit_sha(repo_id, revision):
        bt.logging.warning(f"UID {uid}: {revision} is not a real commit SHA")
        return False
    if not check_duplicate_weights(api, path, uid, submission_times.get(uid, "")):
        return False
    return True


def run_stage1(api, submissions, submission_times, config, all_years, conn, benchmarks, eval_round):
    """Evaluate all miners for leak detection. Returns {uid: score}.

    Downloads for the next miner are prefetched while the current miner is
    validated/evaluated, and models within a miner are loaded in VRAM-sized
    batches and evaluated concurrently on separate CUDA streams.
    """
    WORST_SCORE = 0.0
    leak_scores = {}
    sorted_uids = sorted(submissions.keys(), key=lambda u: submission_times.get(u, "9999"))
    total = len(sorted_uids)
    device = get_device()

    def _fail_repo(uid, repo_str, years):
        for year in years:
            save_result(conn, uid, year, repo_str, False, WORST_SCORE, 0.0, 0.0, eval_round)
            if api.submit_eval_detail(eval_round, uid, year, repo_str, False, WORST_SCORE, 0.0, 0.0):
                mark_synced(conn, uid, year, repo_str, eval_round)

    with ThreadPoolExecutor(1) as dl_pool:
        future = (
            dl_pool.submit(_download_all_for_uid, sorted_uids[0], submissions[sorted_uids[0]], all_years, config, conn)
            if sorted_uids else None
        )

        for i, uid in enumerate(sorted_uids):
            models = submissions[uid]
            bt.logging.info(f"UID {uid}: {len(models)} years submitted")

            year_scores = {year: WORST_SCORE for year in all_years}
            cached_scores, downloaded = future.result()
            year_scores.update(cached_scores)
            if cached_scores:
                bt.logging.info(f"UID {uid}: {len(cached_scores)} scores loaded from cache")

            if i + 1 < len(sorted_uids):
                next_uid = sorted_uids[i + 1]
                future = dl_pool.submit(
                    _download_all_for_uid, next_uid, submissions[next_uid], all_years, config, conn
                )

            # Validate in parallel (CPU-only: SHA hash + duplicate-weight check)
            valid_repos = []
            if downloaded:
                with ThreadPoolExecutor(len(downloaded)) as val_pool:
                    val_futures = {}
                    for repo_str, info in downloaded.items():
                        if "error" in info:
                            bt.logging.error(f"UID {uid}: {repo_str} — {type(info['error']).__name__}")
                            _fail_repo(uid, repo_str, info["years"])
                            continue
                        val_futures[val_pool.submit(
                            _validate_repo, repo_str, info["path"], uid, api, submission_times
                        )] = (repo_str, info)

                    for f in as_completed(val_futures):
                        repo_str, info = val_futures[f]
                        if f.result():
                            valid_repos.append((repo_str, info["path"], info["years"], info["size"]))
                        else:
                            _fail_repo(uid, repo_str, info["years"])
                            shutil.rmtree(info["path"], ignore_errors=True)
                            _release_disk_space(info["size"])

            # Load + eval in VRAM-sized batches
            eval_start = time.time()
            while valid_repos:
                if time.time() - eval_start > config["max_eval_seconds"]:
                    bt.logging.warning(f"UID {uid}: timeout, remaining repos skipped")
                    for _repo_str, path, _years, size in valid_repos:
                        shutil.rmtree(path, ignore_errors=True)
                        _release_disk_space(size)
                    valid_repos = []
                    break

                batch, valid_repos, oversized = _load_batch(valid_repos, device, config)

                for repo_str, years, path, size in oversized:
                    _fail_repo(uid, repo_str, years)
                    shutil.rmtree(path, ignore_errors=True)
                    _release_disk_space(size)

                eval_results = _eval_batch(batch, device, benchmarks)

                for repo_str, year_results in eval_results.items():
                    for year, (passed, score, median_unknown, median_known, failed_leak, passed_known) in year_results.items():
                        year_scores[year] = score
                        save_result(conn, uid, year, repo_str, passed, score, median_unknown, median_known, eval_round)
                        if api.submit_eval_detail(eval_round, uid, year, repo_str, passed, score, median_unknown, median_known):
                            mark_synced(conn, uid, year, repo_str, eval_round)

                        if passed:
                            reason = f" (score={score:.4f})"
                        elif failed_leak is None:
                            reason = " (eval error)"
                        elif failed_leak and not passed_known:
                            reason = f" (leaked future knowledge, unknown={median_unknown:.4f}; failed known facts, known={median_known:.4f})"
                        elif failed_leak:
                            reason = f" (leaked future knowledge, unknown={median_unknown:.4f})"
                        else:
                            reason = f" (failed known facts, known={median_known:.4f})"
                        bt.logging.info(f"UID {uid}: year {year} {'PASSED' if passed else 'FAILED'}{reason}")
                        bt.logging.debug(f"UID {uid} year {year}: unknown={median_unknown:.4f} known={median_known:.4f} score={score:.4f}")

                _free_batch(batch)

            leak_scores[uid] = sum(year_scores.values()) / len(all_years)
            bt.logging.debug(f"UID {uid}: leak_score={leak_scores[uid]:.4f}")
            bt.logging.info(f"Stage 1 progress: {i + 1}/{total} ({(i + 1) / total:.0%})")

    _sync_eval_details(api, conn)
    return leak_scores


def qualify(leak_scores, config):
    """Filter and normalize miners for Stage 2. Returns (qualified, normalized_leak)."""
    top_n = config.get("top_n_for_quality", 10)
    min_eval_score = config.get("min_eval_score", -3.0)
    ranked = sorted(leak_scores.items(), key=lambda x: x[1])
    qualified = [(uid, score) for uid, score in ranked if score < min_eval_score][:top_n]

    bt.logging.info(f"Qualified: {len(qualified)} miners — UIDs: {[uid for uid, _ in qualified]}")

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
            win_rates = run_quality_duels(qualified, submissions, questions, metagraph, all_years)
            leak_weight = config.get("leak_weight", 0.7)
            quality_weight = config.get("quality_weight", 0.3)
            final_scores = np.zeros(metagraph.n)
            for uid, _ in qualified:
                final_scores[uid] = leak_weight * normalized_leak[uid] + quality_weight * win_rates[uid]
                bt.logging.info(f"UID {uid}: final={final_scores[uid]:.4f} (leak={normalized_leak[uid]:.4f} quality={win_rates[uid]:.4f})")

    winner = None
    if final_scores.sum() > 0:
        max_score = final_scores.max()
        tied_uids = [uid for uid in range(metagraph.n) if final_scores[uid] == max_score]
        if len(tied_uids) > 1:
            winner = min(tied_uids, key=lambda u: submission_times.get(u, "9999"))
            bt.logging.info(f"Tie between UIDs {tied_uids}, earliest submission wins")
        else:
            winner = tied_uids[0]
        bt.logging.info(f"Winner: UID {winner} score={final_scores[winner]:.4f}")

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
    bt.logging.set_info()

    api = BackendAPI(BACKEND_URL)

    config = api.get_config()
    ALL_YEARS = api.get_years()
    eval_round = api.get_eval_round()
    bt.logging.info(f"Config: {config}")
    bt.logging.info(f"Eval round: {eval_round}")

    netuid = NETWORKS[args.network]["netuid"]
    owner_uid = NETWORKS[args.network]["owner_uid"]

    conn = get_connection()
    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(netuid=netuid)

    _sync_eval_details(api, conn)

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
    # Preload benchmarks (fail fast if backend is down)
    # =========================================
    benchmarks = _preload_benchmarks(api, ALL_YEARS)
    bt.logging.info(f"Loaded benchmarks for {len(benchmarks)} years")

    # =========================================
    # STAGE 1: Leak detection
    # =========================================
    bt.logging.info("=== Stage 1: Leak detection ===")
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
    bt.logging.info(f"Weights set: {dict(zip(uids, weights))}")

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
