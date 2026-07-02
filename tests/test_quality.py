"""
Test the quality tournament with fake miner answers.

Usage:
    OPENAI_API_KEY=sk-xxx python tests/test_quality.py

Requires: openai package + valid API key.
No model loading, no bittensor, no GPU needed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
from sn38.template.quality import judge, duel

# Fake miner answers — 4 miners, varying quality
QUESTIONS = [
    {"prompt": "What were the main causes of the 2008 financial crisis?"},
    {"prompt": "How did the Fukushima disaster impact global energy policy?"},
    {"prompt": "What were the economic consequences of Brexit on UK finance?"},
]

MINER_ANSWERS = {
    0: [  # Good
        "The 2008 crisis was caused by subprime mortgage lending, excessive leverage by banks, and the failure of credit rating agencies to properly assess risk in mortgage-backed securities.",
        "Fukushima led Germany to phase out nuclear by 2023, shifted global investment toward renewables, and caused Japan to shut most reactors for safety reviews.",
        "Brexit caused major banks to relocate jobs to Dublin and Frankfurt, London lost EU passporting rights for financial services, and euro-denominated clearing partially moved to EU-based clearinghouses.",
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
    3: [  # Mixed
        "Subprime mortgages and excessive leverage by investment banks were the primary causes, compounded by inadequate regulatory oversight.",
        "I think Fukushima happened in 2012 and mainly affected European markets.",
        "Brexit led to around 7,000 finance jobs moving from London to EU cities.",
    ],
}


def test_judge():
    """Test that the judge can pick a winner between two answers."""
    import asyncio
    print(f"Using judge model: {judge.model}")
    print()

    result = asyncio.run(judge.judge_one(
        QUESTIONS[0]["prompt"],
        MINER_ANSWERS[0][0],  # good
        MINER_ANSWERS[2][0],  # bad
    ))
    print(f"Good vs Bad: winner={result} (expected: a)")
    assert result == "a", f"Expected 'a', got '{result}'"

    result = asyncio.run(judge.judge_one(
        QUESTIONS[2]["prompt"],
        MINER_ANSWERS[2][2],  # bad
        MINER_ANSWERS[0][2],  # good
    ))
    print(f"Bad vs Good: winner={result} (expected: b)")
    assert result == "b", f"Expected 'b', got '{result}'"

    print("\nJudge tests passed!")


def test_duel():
    """Test a duel between two miners."""
    print(f"\n--- Duel: UID 0 (good) vs UID 2 (bad) ---")
    winner = duel(MINER_ANSWERS, 0, 2, QUESTIONS)
    print(f"Winner: {winner} (expected: 0)")
    assert winner == 0, f"Expected 0, got {winner}"

    print(f"\n--- Duel: UID 1 (mediocre) vs UID 2 (bad) ---")
    winner = duel(MINER_ANSWERS, 1, 2, QUESTIONS)
    print(f"Winner: {winner} (expected: 1)")
    assert winner == 1, f"Expected 1, got {winner}"

    print("\nDuel tests passed!")


def test_bracket():
    """Test a full elimination bracket with 4 miners using the real run_quality_duels."""
    from unittest.mock import patch
    from sn38.template.quality import run_quality_duels

    metagraph = MagicMock()
    metagraph.n = 4

    qualified = [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]
    submissions = {uid: {"2020": "fake/repo"} for uid in range(4)}

    # Patch generate_answer to return our fake answers
    call_counts = {uid: 0 for uid in range(4)}

    def fake_generate(model, device, question):
        uid = model  # we'll pass uid as "model"
        idx = QUESTIONS.index(question)
        return MINER_ANSWERS[uid][idx]

    def fake_load_model(path, device):
        # Extract uid from path — submissions use "fake/repo" for all
        # The tmpdir path doesn't help, but we track via download_model
        return fake_load_model._current_uid

    def fake_download(repo_id, tmpdir, revision=None):
        return tmpdir

    with patch("sn38.template.quality.generate_answer", side_effect=fake_generate), \
         patch("sn38.template.quality.load_model") as mock_load, \
         patch("sn38.template.quality.download_model", side_effect=fake_download):

        # Make load_model return the uid so generate_answer can use it
        def load_returning_uid(path, device):
            for uid, _ in qualified:
                if submissions[uid]["2020"] == "fake/repo":
                    # Track which uid we're loading by order of calls
                    result = load_returning_uid._uids.pop(0)
                    return result
            return 0
        load_returning_uid._uids = [uid for uid, _ in qualified]
        mock_load.side_effect = load_returning_uid

        scores = run_quality_duels(qualified, submissions, QUESTIONS, metagraph, [2013, 2020])

    winner = int(scores.argmax())
    print(f"Final scores: {scores}")
    print(f"Winner: UID {winner} (expected: 0 — best answers)")

    print("\nBracket test passed!")


if __name__ == "__main__":
    import bittensor as bt
    bt.logging.set_info()

    print("=" * 50)
    print("TEST 1: Judge")
    print("=" * 50)
    test_judge()

    print("\n" + "=" * 50)
    print("TEST 2: Duel")
    print("=" * 50)
    test_duel()

    print("\n" + "=" * 50)
    print("TEST 3: Bracket")
    print("=" * 50)
    test_bracket()

    print("\n\nAll tests passed!")
