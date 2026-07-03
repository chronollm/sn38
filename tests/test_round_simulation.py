"""
Full round simulation: Stage 1 → qualification → Stage 2 → winner.

Usage:
    python -m pytest tests/test_stage1_simulation.py -v
"""

import os
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test")

import numpy as np
import pytest
from sn38.neurons.validator import run_stage1, run_stage2_and_score

VALID_SHA = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
YEARS = [2013, 2014, 2015]
CONFIG = {
    "max_model_bytes": 10**10,
    "max_parameters": 10**10,
    "max_eval_seconds": 999,
    "min_eval_score": -3.0,
    "leak_epsilon": -6.0,
    "top_n_for_quality": 10,
    "leak_weight": 0.7,
    "quality_weight": 0.3,
    "emission_pct": 0.30,
    "owner_uid": 0,
}

# evaluate returns (ratio > threshold, median)
# Called twice per year: unknown then known
PROFILES = {
    "good_strong": [(False, -8.0), (True, -2.0)] * 10,      # score = -6.0
    "good_medium": [(False, -7.5), (True, -4.0)] * 10,      # score = -3.5
    "good_weak":   [(False, -7.0), (True, -4.5)] * 10,      # score = -2.5 (won't qualify)
    "leaker":      [(True, -3.0), (True, -2.0)] * 10,       # WORST_SCORE
    "lobotomized": [(False, -9.0), (False, -9.0)] * 10,     # WORST_SCORE
    "copier":      [(False, -8.0), (True, -2.0)] * 10,      # duplicate → WORST_SCORE
}

MINERS = {
    1: ("good_strong", "user/strong"),
    2: ("good_medium", "user/medium"),
    3: ("good_weak",   "user/weak"),
    4: ("leaker",      "user/leaker"),
    5: ("lobotomized", "user/lobotomized"),
    6: ("copier",      "user/strong"),  # same repo as miner 1
}

SUBMISSION_TIMES = {
    1: "2026-07-01T08:00:00",
    2: "2026-07-01T09:00:00",
    3: "2026-07-01T10:00:00",
    4: "2026-07-01T11:00:00",
    5: "2026-07-01T12:00:00",
    6: "2026-07-01T13:00:00",
}

SUBMISSIONS = {
    uid: {str(y): f"user/m-{uid}@{VALID_SHA}" for y in YEARS}
    for uid in MINERS
}


@pytest.fixture
def stage1_patched(monkeypatch, tmp_path):
    """Patch Stage 1 dependencies."""
    for uid, (profile, repo) in MINERS.items():
        d = tmp_path / str(uid)
        d.mkdir(exist_ok=True)
        (d / "model.safetensors").write_bytes(f"weights_{repo}".encode())

    claimed_hashes = {}

    def fake_download(repo_id, local_dir, revision=None):
        uid = int(repo_id.split("-")[1])
        return str(tmp_path / str(uid))

    def fake_check(api, path, uid, snapshot_at):
        import hashlib
        h = hashlib.sha256(open(os.path.join(path, "model.safetensors"), "rb").read()).hexdigest()
        if h in claimed_hashes and claimed_hashes[h] != uid:
            return False
        claimed_hashes[h] = uid
        return True

    eval_iterators = {uid: iter(PROFILES[profile]) for uid, (profile, _) in MINERS.items()}
    current_uid = [None]

    def fake_load(path, device):
        current_uid[0] = int(os.path.basename(path))
        return MagicMock()

    def fake_evaluate(model, device, benchmark):
        return next(eval_iterators[current_uid[0]])

    monkeypatch.setattr("sn38.neurons.validator.download_model", fake_download)
    monkeypatch.setattr("sn38.neurons.validator.check_duplicate_weights", fake_check)
    monkeypatch.setattr("sn38.neurons.validator.load_model", fake_load)
    monkeypatch.setattr("sn38.neurons.validator.count_model_params", lambda m: 1_000_000)
    monkeypatch.setattr("sn38.neurons.validator.get_repo_file_size", lambda r, v: 1000)
    monkeypatch.setattr("sn38.neurons.validator.get_device", lambda: "cpu")
    monkeypatch.setattr("sn38.neurons.validator.evaluate", fake_evaluate)
    monkeypatch.setattr("sn38.neurons.validator.get_cached_result", lambda c, u, y, r: None)
    monkeypatch.setattr("sn38.neurons.validator.save_result", lambda *a: None)


@pytest.fixture
def leak_scores(stage1_patched):
    """Run Stage 1 and return scores."""
    return run_stage1(MagicMock(), SUBMISSIONS, SUBMISSION_TIMES, CONFIG, YEARS, conn=MagicMock())


# ─── Stage 1 ───

def test_good_miners_score_negative(leak_scores):
    assert leak_scores[1] < 0
    assert leak_scores[2] < 0

def test_strong_beats_medium(leak_scores):
    assert leak_scores[1] < leak_scores[2]

def test_weak_above_threshold(leak_scores):
    """good_weak scores -2.5, above min_eval_score (-3.0), won't qualify."""
    assert leak_scores[3] > CONFIG["min_eval_score"]

def test_bad_miners_get_worst(leak_scores):
    assert leak_scores[4] == 0.0  # leaker
    assert leak_scores[5] == 0.0  # lobotomized
    assert leak_scores[6] == 0.0  # copier


# ─── Qualification + Stage 2 ───

@pytest.fixture
def run_stage2(leak_scores, monkeypatch):
    """Factory fixture to run Stage 2 with optional overrides."""
    def _run(win_rates=None, config_override=None):
        cfg = {**CONFIG, **(config_override or {})}
        monkeypatch.setattr("sn38.neurons.validator.run_quality_duels",
                            lambda *a: win_rates if win_rates is not None else np.zeros(10))
        metagraph = MagicMock()
        metagraph.n = 10
        return run_stage2_and_score(
            MagicMock(), leak_scores, SUBMISSIONS, SUBMISSION_TIMES, cfg, YEARS, metagraph
        )
    return _run


def test_only_good_miners_qualify(run_stage2):
    final_scores, winner, uids, weights = run_stage2()
    assert final_scores[1] > 0
    assert final_scores[2] > 0
    assert final_scores[3] == 0  # not qualified (-2.5 > -3.0)
    assert final_scores[4] == 0
    assert final_scores[5] == 0
    assert final_scores[6] == 0


def test_strong_miner_wins(run_stage2):
    _, winner, _, _ = run_stage2()
    assert winner == 1


def test_quality_contributes_to_final_score(run_stage2):
    win_rates_high = np.zeros(10)
    win_rates_high[2] = 1.0
    final_high, _, _, _ = run_stage2(win_rates=win_rates_high)
    final_zero, _, _, _ = run_stage2(win_rates=np.zeros(10))
    assert final_high[2] > final_zero[2]


def test_no_qualified_returns_no_winner(run_stage2):
    final_scores, winner, _, _ = run_stage2(config_override={"min_eval_score": -100.0})
    assert winner is None
    assert final_scores.sum() == 0


# ─── Emission split ───

def test_emission_split_matches_config(run_stage2):
    emission_pct = CONFIG["emission_pct"]
    _, winner, uids, weights = run_stage2()
    assert weights[uids.index(winner)] == emission_pct
    assert weights[uids.index(0)] == 1.0 - emission_pct
    assert abs(sum(weights) - 1.0) < 1e-9

def test_no_qualified_burns_to_owner(run_stage2):
    """No qualified miners → all emissions go to owner (burn)."""
    _, winner, uids, weights = run_stage2(config_override={"min_eval_score": -100.0})
    assert winner is None
    assert uids == [0]
    assert weights == [1.0]
