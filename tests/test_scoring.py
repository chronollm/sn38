"""
Test scoring normalization and edge cases.

Usage:
    python -m pytest tests/test_scoring.py -v
"""


def normalize(score, eval_threshold=-3.0, eval_best=-6.0):
    """Same formula as validator.py"""
    return max(0.0, min(1.0, (eval_threshold - score) / (eval_threshold - eval_best)))


# ─── Normalization direction ───

def test_more_negative_is_higher():
    assert normalize(-6.0) > normalize(-4.0)
    assert normalize(-4.0) > normalize(-3.0)

def test_best_score_is_one():
    assert normalize(-6.0) == 1.0

def test_threshold_score_is_zero():
    assert normalize(-3.0) == 0.0

def test_middle_score():
    assert normalize(-4.5) == 0.5


# ─── Clamping ───

def test_worst_score_clamps_to_zero():
    assert normalize(0.0) == 0.0

def test_better_than_best_clamps_to_one():
    assert normalize(-8.0) == 1.0
    assert normalize(-100.0) == 1.0

def test_positive_score_clamps_to_zero():
    assert normalize(1.0) == 0.0
    assert normalize(10.0) == 0.0


# ─── WORST_SCORE must not be rewarded ───

def test_worst_score_not_rewarded():
    WORST_SCORE = 0.0
    good_score = -4.0
    assert normalize(good_score) > normalize(WORST_SCORE)

def test_missing_years_not_rewarded():
    """A miner with missing years (WORST_SCORE=0.0) must score lower."""
    WORST_SCORE = 0.0
    good_years = [-3.5, -4.0, -3.8, -4.2]
    missing_years = [-3.5, -4.0, WORST_SCORE, WORST_SCORE]

    avg_good = sum(good_years) / len(good_years)
    avg_missing = sum(missing_years) / len(missing_years)

    assert normalize(avg_good) > normalize(avg_missing)


# ─── Config robustness ───

def test_swapped_config_still_works():
    """Even if someone swaps threshold and best, direction stays correct."""
    n = lambda s: normalize(s, eval_threshold=-6.0, eval_best=-3.0)
    assert n(-6.0) == 0.0
    assert n(-3.0) == 1.0
    assert n(-4.5) == 0.5


# ─── Score formula ───

def test_score_formula_good_model():
    """Good model: doesn't leak (unknown very negative) + knows training data (known close to 0)."""
    median_unknown = -8.0
    median_known = -2.0
    score = median_unknown - median_known
    assert score == -6.0
    assert normalize(score) == 1.0

def test_score_formula_lobotomized():
    """Lobotomized model: doesn't leak but doesn't know anything either."""
    median_unknown = -9.0
    median_known = -9.0
    score = median_unknown - median_known
    assert score == 0.0
    assert normalize(score) == 0.0

def test_score_formula_leaker():
    """Leaker: knows future but also knows training data."""
    median_unknown = -2.0
    median_known = -2.0
    score = median_unknown - median_known
    assert score == 0.0
    assert normalize(score) == 0.0

def test_good_model_beats_lobotomized():
    good = -8.0 - (-2.0)     # -6.0
    lobotomized = -9.0 - (-9.0)  # 0.0
    assert normalize(good) > normalize(lobotomized)

def test_good_model_beats_leaker():
    good = -8.0 - (-2.0)    # -6.0
    leaker = -2.0 - (-2.0)  # 0.0
    assert normalize(good) > normalize(leaker)
