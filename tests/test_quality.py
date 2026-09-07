"""
Test per-prompt win rate scoring by replaying actual round duel results.

Usage:
    python -m pytest tests/test_quality.py -v
"""

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")

import numpy as np
import pytest

from sn38.template.quality import _run_round_robin

UIDS = [130, 124, 140, 181, 131, 187, 151]
N_PROMPTS = 104

# Round 8 actual duel results: (uid_a, wins_a, uid_b, wins_b)
ROUND8_DUELS = [
    (130, 69, 124, 19),
    (130, 48, 140, 34),
    (130, 97, 181, 7),
    (130, 40, 131, 37),
    (130, 95, 187, 8),
    (130, 68, 151, 19),
    (124, 25, 140, 65),
    (124, 82, 181, 15),
    (124, 11, 131, 78),
    (124, 85, 187, 15),
    (124, 38, 151, 44),
    (140, 98, 181, 5),
    (140, 31, 131, 41),
    (140, 94, 187, 9),
    (140, 66, 151, 24),
    (181, 2, 131, 101),
    (181, 13, 187, 5),
    (181, 19, 151, 81),
    (131, 101, 187, 2),
    (131, 70, 151, 18),
    (187, 16, 151, 85),
]


def _make_metagraph(n=256):
    mg = MagicMock()
    mg.n = n
    return mg


@pytest.fixture
def round8_win_rates():
    duel_iter = iter(ROUND8_DUELS)

    def fake_duel(miner_completions, uid_a, uid_b, prompts):
        expected = next(duel_iter)
        assert (uid_a, uid_b) == (expected[0], expected[2])
        return expected[1], expected[3], N_PROMPTS

    completions = {uid: ["c"] * N_PROMPTS for uid in UIDS}
    mg = _make_metagraph()

    with patch("sn38.template.quality.duel", side_effect=fake_duel):
        return _run_round_robin(completions, [{"prompt": "p"}] * N_PROMPTS, mg, UIDS)


# ─── Per-prompt ranking ───

def test_per_prompt_ranking_top2(round8_win_rates):
    wr = round8_win_rates
    ranked = sorted(UIDS, key=lambda u: -wr[u])
    assert ranked[0] == 131
    assert ranked[1] == 130


def test_per_prompt_full_ranking(round8_win_rates):
    wr = round8_win_rates
    ranked = sorted(UIDS, key=lambda u: -wr[u])
    assert ranked == [131, 130, 140, 151, 124, 181, 187]


def test_no_zero_for_losers(round8_win_rates):
    wr = round8_win_rates
    for uid in UIDS:
        assert wr[uid] > 0


def test_win_rates_between_zero_and_one(round8_win_rates):
    wr = round8_win_rates
    for uid in UIDS:
        assert 0 <= wr[uid] <= 1.0


# ─── Binary scoring would differ ───

def test_binary_ranking_differs():
    binary_wins = {uid: 0 for uid in UIDS}
    for uid_a, wins_a, uid_b, wins_b in ROUND8_DUELS:
        if wins_a > wins_b:
            binary_wins[uid_a] += 1
        elif wins_b > wins_a:
            binary_wins[uid_b] += 1
    binary_ranked = sorted(UIDS, key=lambda u: -binary_wins[u])
    assert binary_ranked[0] == 130
    assert binary_ranked[1] == 131


# ─── Round 9: only 2 miners qualified ───

ROUND9_UIDS = [140, 131]
ROUND9_N_PROMPTS = 401

ROUND9_DUELS = [
    (140, 120, 131, 173),
]


@pytest.fixture
def round9_win_rates():
    duel_iter = iter(ROUND9_DUELS)

    def fake_duel(miner_completions, uid_a, uid_b, prompts):
        expected = next(duel_iter)
        assert (uid_a, uid_b) == (expected[0], expected[2])
        return expected[1], expected[3], ROUND9_N_PROMPTS

    completions = {uid: ["c"] * ROUND9_N_PROMPTS for uid in ROUND9_UIDS}
    mg = _make_metagraph()

    with patch("sn38.template.quality.duel", side_effect=fake_duel):
        return _run_round_robin(completions, [{"prompt": "p"}] * ROUND9_N_PROMPTS, mg, ROUND9_UIDS)


def test_round9_loser_not_zero(round9_win_rates):
    """With binary scoring UID 140 got 0.0 despite winning 120/401 prompts."""
    wr = round9_win_rates
    assert wr[140] == pytest.approx(120 / 401)
    assert wr[131] == pytest.approx(173 / 401)


def test_round9_ranking(round9_win_rates):
    wr = round9_win_rates
    assert wr[131] > wr[140]


# ─── Edge cases ───

@patch("sn38.template.quality.duel")
def test_two_miners_loser_gets_credit(mock_duel):
    mock_duel.return_value = (60, 40, 100)
    mg = _make_metagraph()
    wr = _run_round_robin({1: ["c"] * 100, 2: ["c"] * 100}, [{"prompt": "p"}] * 100, mg, [1, 2])
    assert wr[1] == pytest.approx(0.6)
    assert wr[2] == pytest.approx(0.4)


@patch("sn38.template.quality.duel")
def test_all_ties(mock_duel):
    mock_duel.return_value = (0, 0, 100)
    mg = _make_metagraph()
    wr = _run_round_robin({1: ["c"] * 100, 2: ["c"] * 100}, [{"prompt": "p"}] * 100, mg, [1, 2])
    assert wr[1] == 0.0
    assert wr[2] == 0.0