"""Honest out-of-sample test: fit on the past, predict the future."""
import numpy as np
import pandas as pd

from .data import competition_tier, load_results
from .elo import run_elo
from .model import Model, Settings


def score(probs, outcome):
    """probs: (n, 3) for [home win, draw, away win]; outcome: 0/1/2."""
    n = len(outcome)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcome] = 1
    p_true = np.clip(probs[np.arange(n), outcome], 1e-12, 1)
    cum_p, cum_o = np.cumsum(probs, 1)[:, :2], np.cumsum(onehot, 1)[:, :2]
    return {
        "accuracy": float(np.mean(probs.argmax(1) == outcome)),
        "log_loss": float(-np.mean(np.log(p_true))),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, 1))),
        "rps": float(np.mean(np.sum((cum_p - cum_o) ** 2, 1) / 2)),
    }


def run_backtest(start="2022-01-01", end=None, refit_days=182, settings=None, verbose=True,
                 df=None, elo_fn=run_elo, major_tiers=("world_cup", "continental")):
    df = load_results() if df is None else df
    if end is not None:
        df = df[df["date"] < pd.Timestamp(end)].reset_index(drop=True)
    _, pre_h, pre_a = elo_fn(df)  # pre-match Elo only uses earlier results
    start = pd.Timestamp(start)
    test_idx = np.where(df["date"] >= start)[0]
    outcome = np.where(df["home_score"] > df["away_score"], 0,
                       np.where(df["home_score"] == df["away_score"], 1, 2))

    variants = {
        "Full model (Elo + Dixon-Coles)": settings or Settings(),
        "Elo only": Settings(use_team_strengths=False, window_years=(settings or Settings()).window_years),
    }
    preds = {k: np.zeros((len(test_idx), 3)) for k in variants}
    cut = start
    end = df["date"].max() + pd.Timedelta(days=1)
    while cut < end:
        nxt = cut + pd.Timedelta(days=refit_days)
        chunk = [k for k, i in enumerate(test_idx) if cut <= df["date"].iat[i] < nxt]
        if chunk:
            for name, s in variants.items():
                m = Model.fit(df, cutoff=cut, settings=s, pre_elo=(pre_h, pre_a), final_elo={})
                for k in chunk:
                    i = test_idx[k]
                    preds[name][k] = m.outcome_probs(
                        df["home_team"].iat[i], df["away_team"].iat[i], bool(df["neutral"].iat[i]),
                        elo_home=pre_h[i], elo_away=pre_a[i])
        cut = nxt

    y = outcome[test_idx]
    train = df["date"] < start
    base = np.bincount(outcome[train.to_numpy()], minlength=3) / train.sum()
    preds["Base rates (no model)"] = np.tile(base, (len(test_idx), 1))

    tiers = df["tournament"].iloc[test_idx].map(competition_tier).to_numpy()
    major = np.isin(tiers, list(major_tiers))
    results = {}
    for name, p in preds.items():
        results[name] = {"all": score(p, y)}
        if major.any():
            results[name]["major_finals"] = score(p[major], y[major])

    if verbose:
        print(f"Backtest: {len(test_idx)} matches from {start.date()} to {df['date'].max().date()}"
              f" (refit every {refit_days} days)\n")
        for subset in (["all", "major_finals"] if major.any() else ["all"]):
            print("All matches" if subset == "all" else "World Cup & continental finals only")
            print(f"  {'model':32s} {'accuracy':>9s} {'log loss':>9s} {'brier':>7s} {'RPS':>7s}")
            for name, r in results.items():
                s = r[subset]
                print(f"  {name:32s} {100 * s['accuracy']:8.1f}% {s['log_loss']:9.3f}"
                      f" {s['brier']:7.3f} {s['rps']:7.4f}")
            print()
        print("Lower log loss / Brier / RPS is better.")
    return results
