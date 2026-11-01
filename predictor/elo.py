"""World Football Elo style ratings."""
from collections import defaultdict

import numpy as np

from .data import ELO_K, competition_tier

START = 1500.0
HOME_BONUS = 100.0


def goal_multiplier(margin: int) -> float:
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return 1.75 + (margin - 3) / 8


def expected(diff: float) -> float:
    """Win expectancy for a rating gap (home bonus already included)."""
    return 1.0 / (10 ** (-diff / 400) + 1)


def run_elo(df):
    """Run through every match in date order.

    Returns (final ratings, pre-match home Elo array, pre-match away Elo array).
    """
    ratings = defaultdict(lambda: START)
    pre_h = np.empty(len(df))
    pre_a = np.empty(len(df))
    cols = zip(df["home_team"], df["away_team"], df["home_score"],
               df["away_score"], df["tournament"], df["neutral"])
    for i, (h, a, hs, as_, tour, neutral) in enumerate(cols):
        rh, ra = ratings[h], ratings[a]
        pre_h[i], pre_a[i] = rh, ra
        diff = rh - ra + (0 if neutral else HOME_BONUS)
        we = expected(diff)
        w = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        k = ELO_K[competition_tier(tour)] * goal_multiplier(hs - as_)
        change = k * (w - we)
        ratings[h] = rh + change
        ratings[a] = ra - change
    return dict(ratings), pre_h, pre_a
