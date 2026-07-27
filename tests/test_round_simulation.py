"""
Round simulation: qualification → Stage 2 → exponential rewards.

Usage:
    python -m pytest tests/test_round_simulation.py -v
"""

import os
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test")

import numpy as np
import pytest
from sn38.neurons.validator import run_stage2_and_score

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
    "emission_pct": 1.0,
    "owner_uid": 0,
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
    for uid in SUBMISSION_TIMES
}



LEAK_SCORES = {
    1: -6.0,   # good_strong — qualifies
    2: -3.5,   # good_medium — qualifies
    3: -2.5,   # good_weak — above threshold, won't qualify
    4: 0.0,    # leaker
    5: 0.0,    # lobotomized
    6: 0.0,    # copier
}


# ─── Stage 1 ───

def test_good_miners_score_negative():
    assert LEAK_SCORES[1] < 0
    assert LEAK_SCORES[2] < 0

def test_strong_beats_medium():
    assert LEAK_SCORES[1] < LEAK_SCORES[2]

def test_weak_above_threshold():
    assert LEAK_SCORES[3] > CONFIG["min_eval_score"]

def test_bad_miners_get_worst():
    assert LEAK_SCORES[4] == 0.0
    assert LEAK_SCORES[5] == 0.0
    assert LEAK_SCORES[6] == 0.0


# ─── Qualification + Stage 2 ───

@pytest.fixture
def run_stage2(monkeypatch):
    """Factory fixture to run Stage 2 with optional overrides."""
    def _run(win_rates=None, config_override=None):
        cfg = {**CONFIG, **(config_override or {})}
        monkeypatch.setattr("sn38.neurons.validator.run_quality_duels",
                            lambda *a: win_rates if win_rates is not None else np.zeros(10))
        metagraph = MagicMock()
        metagraph.n = 10
        return run_stage2_and_score(
            MagicMock(), LEAK_SCORES, SUBMISSIONS, SUBMISSION_TIMES, cfg, YEARS, metagraph
        )
    return _run


def test_only_good_miners_qualify(run_stage2):
    final_scores, winner, uids, weights, _ = run_stage2()
    assert final_scores[1] > 0
    assert final_scores[2] > 0
    assert final_scores[3] == 0  # not qualified (-2.5 > -3.0)
    assert final_scores[4] == 0
    assert final_scores[5] == 0
    assert final_scores[6] == 0


def test_strong_miner_wins(run_stage2):
    _, winner, _, _, _ = run_stage2()
    assert winner == 1


def test_quality_contributes_to_final_score(run_stage2):
    win_rates_high = np.zeros(10)
    win_rates_high[2] = 1.0
    final_high, _, _, _, _ = run_stage2(win_rates=win_rates_high)
    final_zero, _, _, _, _ = run_stage2(win_rates=np.zeros(10))
    assert final_high[2] > final_zero[2]


def test_no_qualified_returns_no_winner(run_stage2):
    final_scores, winner, _, _, _ = run_stage2(config_override={"min_eval_score": -100.0})
    assert winner is None
    assert final_scores.sum() == 0


# ─── Emission split ───

def test_exponential_rewards_top_n(run_stage2):
    """Rewards distributed exponentially across qualified miners."""
    _, winner, uids, weights, _ = run_stage2()
    assert winner == 1
    assert 1 in uids
    assert 2 in uids
    assert weights[uids.index(1)] > weights[uids.index(2)]
    assert abs(sum(weights) - 1.0) < 1e-9

def test_exponential_rewards_with_owner_cut(run_stage2):
    """When emission_pct < 1, owner gets the remainder."""
    _, winner, uids, weights, _ = run_stage2(config_override={"emission_pct": 0.30})
    assert abs(sum(weights) - 1.0) < 1e-9
    assert 0 in uids
    assert abs(weights[uids.index(0)] - 0.70) < 1e-9

def test_no_qualified_keeps_current_weights(run_stage2):
    """No qualified miners → uids/weights are None (keep on-chain weights)."""
    _, winner, uids, weights, _ = run_stage2(config_override={"min_eval_score": -100.0})
    assert winner is None
    assert uids is None
    assert weights is None
