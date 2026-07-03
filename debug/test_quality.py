"""
Test the quality tournament with mocked judge.

Usage:
    python -m pytest tests/test_quality.py -v
"""

import os
os.environ.setdefault("OPENAI_API_KEY", "test")

from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from sn38.template.quality import judge, duel, run_quality_duels

QUESTIONS = [
    {"prompt": "What were the main causes of the 2008 financial crisis?"},
    {"prompt": "How did the Fukushima disaster impact global energy policy?"},
    {"prompt": "What were the economic consequences of Brexit on UK finance?"},
]

MINER_ANSWERS = {
    0: [  # Good
        "The 2008 crisis was caused by subprime mortgage lending and excessive leverage.",
        "Fukushima led Germany to phase out nuclear and shifted investment toward renewables.",
        "Brexit caused major banks to relocate jobs to Dublin and Frankfurt.",
    ],
    1: [  # Mediocre
        "Banks lent too much money to people who couldn't pay it back.",
        "Some countries stopped using nuclear energy after Fukushima.",
        "Brexit was bad for UK banks, some moved to Europe.",
    ],
    2: [  # Bad
        "The crisis was caused by the Euro being too strong.",
        "Fukushima increased oil production in the Middle East.",
        "Brexit had minimal impact because London adapted quickly.",
    ],
}


@pytest.fixture(autouse=True)
def mock_judge(monkeypatch):
    """Mock the judge to pick the longer, more detailed answer."""
    async def fake_judge_one(self, question, answer_a, answer_b):
        if len(answer_a) > len(answer_b):
            return "a"
        elif len(answer_b) > len(answer_a):
            return "b"
        return "tie"

    monkeypatch.setattr("sn38.template.quality.Judge.judge_one", fake_judge_one)


def test_judge_picks_better_answer():
    import asyncio
    result = asyncio.run(judge.judge_one(
        QUESTIONS[0]["prompt"],
        MINER_ANSWERS[0][0],  # longer, more detailed
        MINER_ANSWERS[2][0],  # shorter
    ))
    assert result == "a"


def test_judge_reversed_order():
    import asyncio
    result = asyncio.run(judge.judge_one(
        QUESTIONS[0]["prompt"],
        MINER_ANSWERS[2][0],  # shorter
        MINER_ANSWERS[0][0],  # longer
    ))
    assert result == "b"


def test_duel_good_beats_bad():
    winner = duel(MINER_ANSWERS, 0, 2, QUESTIONS)
    assert winner == 0


def test_duel_good_beats_mediocre():
    winner = duel(MINER_ANSWERS, 0, 1, QUESTIONS)
    assert winner == 0


def test_bracket():
    metagraph = MagicMock()
    metagraph.n = 3

    qualified = [(0, 1.0), (1, 1.0), (2, 1.0)]
    submissions = {uid: {"2013": "fake/repo", "2020": "fake/repo"} for uid in range(3)}

    def fake_generate(model, device, prompt):
        uid = model
        idx = next(i for i, q in enumerate(QUESTIONS) if q["prompt"] == prompt)
        return MINER_ANSWERS[uid][idx]

    uids_iter = iter([0, 1, 2] * 2)  # called once per year per miner

    with patch("sn38.template.quality.generate_answer", side_effect=fake_generate), \
         patch("sn38.template.quality.load_model", side_effect=lambda p, d: next(uids_iter)), \
         patch("sn38.template.quality.download_model", side_effect=lambda r, t, revision=None: t):

        scores = run_quality_duels(qualified, submissions, QUESTIONS, metagraph, [2013, 2020])

    assert scores[0] > scores[1]  # good beats mediocre
    assert scores[0] > scores[2]  # good beats bad
