"""
Live smoke test for the parallel Stage 1 implementation (feat/eval-speedup).

Runs the real run_stage1() against every miner in a round (round 2 by
default), or a specific subset via --uids. Also logs the earliest two miners
in the biggest duplicate-weight cluster in the round, to make it easy to spot
check_duplicate_weights() correctly rejecting the copier in the output.

Downloads real models from HuggingFace and runs real inference — needs a
GPU (or at least enough RAM/time on CPU) and network access. Does not touch
subtensor/wallet/weight-setting, only Stage 1 scoring.

/benchmark and /models/check-hash on the real backend require TEE-attested
mTLS (see sn38/template/tee.py), which isn't available outside dstack, so
this script serves those two locally instead:
  - benchmark items come from a local leak_events.csv (same schema/known-leak
    split the real backend uses — year <= cutoff is "known", year > cutoff
    is "leak"), via --leak-csv
  - duplicate-weight checks are a naive in-process first-claim-wins map,
    since the real backend's cross-round history can't be replicated locally

/config, /years and /submissions are public and still hit the real backend.

Usage:
    python scripts/test_stage1_live.py --leak-csv /path/to/leak_events.csv \
        [--round 2] [--uids 90,91]
"""

import argparse
import collections
import csv
import functools
import shutil

import bittensor as bt
import tqdm
from sn38.neurons.validator import _preload_benchmarks, run_stage1
from sn38.template.backend_api import BackendAPI
from sn38.template.validator_db import get_connection

BACKEND_URL = "https://api.chronollm.com"


TQDM_MININTERVAL = 10


def tqdm_init():
    original = tqdm.tqdm.__init__

    def patched(*args, **kwargs):
        mininterval = kwargs.get("mininterval")
        if mininterval is None or not isinstance(mininterval, (int, float)) or mininterval < TQDM_MININTERVAL:
            kwargs = kwargs.copy()
            kwargs["mininterval"] = TQDM_MININTERVAL

        return original(*args, **kwargs)

    tqdm.tqdm.__init__ = patched


def tqdm_display():

    def tqdm_display(self, msg=None, pos=None):
        if pos is None:
            pos = abs(self.pos)

        if not msg:
            msg = str(self).replace("|| ", " - ")

        print(f"[tqdm:{pos}] {msg}\n", end="", flush=True)

    # shutil's own fallback (used when it can't detect a real terminal, e.g.
    # under Jupyter Lab) is already 80, so this only ever narrows a width
    # tqdm's auto-detect guessed too wide, never a genuinely narrow terminal.
    ncols = min(shutil.get_terminal_size(fallback=(80, 24)).columns, 80)

    tqdm.tqdm.__init__ = functools.partialmethod(
        tqdm.tqdm.__init__,
        bar_format='{l_bar}{r_bar}',
        ncols=ncols,
    )

    tqdm.tqdm.display = tqdm_display


tqdm_init()
tqdm_display()


def _extract_prompt(sentence, phrase):
    """Mirrors the real backend's benchmark.py:_extract_prompt (rfind, not find,
    in case the phrase recurs earlier in the sentence)."""
    idx = sentence.lower().rfind(phrase.lower())
    if idx < 0:
        return sentence
    return sentence[:idx].rstrip()


def load_leak_items(csv_path):
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sentence, phrase = row["sentence"], row["phrase"]
            items.append({"year": int(row["year"]), "phrase": phrase, "prompt": _extract_prompt(sentence, phrase)})
    return items


class LocalBenchmarkAPI(BackendAPI):
    """Real BackendAPI for the public endpoints, local benchmark/dedup for the
    two TEE-gated ones so Stage 1 can run outside dstack."""

    def __init__(self, backend_url, leak_csv):
        super().__init__(backend_url)
        self._leak_items = load_leak_items(leak_csv)
        self._claimed_hashes = {}

    def get_benchmark(self, cutoff_year, known=False):
        items = [i for i in self._leak_items if (i["year"] <= cutoff_year) == known]
        return {"items": items, "threshold": 0.70 if known else 0.10, "epsilon": -6.0}

    def check_hash(self, weight_hash, uid, snapshot_at):
        owner = self._claimed_hashes.get(weight_hash)
        if owner is not None and owner != uid:
            return {"allowed": False, "owner_uid": owner}
        self._claimed_hashes[weight_hash] = uid
        return {"allowed": True}


def find_duplicate_pair(submissions, submission_times):
    """Return the two earliest UIDs sharing the largest identical-repo cluster."""
    repo_to_uids = collections.defaultdict(set)
    for uid, info in submissions.items():
        for repo in info.values():
            repo_to_uids[repo].add(uid)

    clusters = [uids for uids in repo_to_uids.values() if len(uids) > 1]
    if not clusters:
        return []

    biggest = max(clusters, key=len)
    ordered = sorted(biggest, key=lambda u: submission_times.get(u, "9999"))
    return ordered[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leak-csv", required=True, help="Path to leak_events.csv")
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--uids", type=str, default=None,
                        help="Comma-separated UIDs to test. Omit to test every miner in the round")
    args = parser.parse_args()

    bt.logging.set_info()
    api = LocalBenchmarkAPI(BACKEND_URL, args.leak_csv)
    conn = get_connection()

    config = api.get_config()
    all_years = api.get_years()

    all_submissions, submission_times = api.get_submissions(args.round)

    dupe_uids = find_duplicate_pair(all_submissions, submission_times)
    if dupe_uids:
        bt.logging.info(f"Duplicate-weight cluster candidates: {dupe_uids}")
    else:
        bt.logging.warning("No duplicate-weight cluster found in this round")

    if args.uids is None:
        submissions = all_submissions
    else:
        test_uids = {int(u) for u in args.uids.split(",")} | set(dupe_uids)
        missing = test_uids - all_submissions.keys()
        if missing:
            bt.logging.warning(f"UIDs not present in round {args.round}: {missing}")
        submissions = {uid: all_submissions[uid] for uid in test_uids if uid in all_submissions}

    bt.logging.info(f"Testing {len(submissions)} UIDs: {sorted(submissions.keys())}")

    benchmarks = _preload_benchmarks(api, all_years)

    leak_scores = run_stage1(
        api, submissions, submission_times, config, all_years, conn,
        benchmarks, args.round,
    )

    bt.logging.info("=== Results ===")
    for uid, score in sorted(leak_scores.items()):
        bt.logging.info(f"UID {uid}: leak_score={score:.4f}")


if __name__ == "__main__":
    main()
