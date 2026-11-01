"""Walk-forward evaluation: refit on the past, predict the next block of games, repeat.

Every prediction only uses matches (and player data) from before that game.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from .backtest import score
from .model import MAX_GOALS, Model, elo_gap


def rates_vec(m: Model, d, pre_h, pre_a):
    """Expected goals for many matches at once."""
    gap = elo_gap(pre_h, pre_a)
    att = d["home_team"].map(m.attack).fillna(0).to_numpy()
    att_a = d["away_team"].map(m.attack).fillna(0).to_numpy()
    dfn = d["home_team"].map(m.defence).fillna(0).to_numpy()
    dfn_a = d["away_team"].map(m.defence).fillna(0).to_numpy()
    xg = np.zeros(len(d))
    for name, wt in m.gamma.items():
        xg += wt * np.nan_to_num(d[name].to_numpy(float))
    h = d["tournament"].map(m.home_league).fillna(m.home_adv).to_numpy() if m.home_league \
        else np.full(len(d), m.home_adv)
    home = (~d["neutral"]).to_numpy(float)
    lh = np.exp(m.mu + h * home + att - dfn_a + m.beta * gap + xg)
    la = np.exp(m.mu + att_a - dfn - m.beta * gap - xg)
    return lh, la


def outcome_vec(lh, la, rho, draw_boost=0.0):
    k = np.arange(MAX_GOALS + 1)
    g = poisson.pmf(k[None, :], lh[:, None])[:, :, None] * poisson.pmf(k[None, :], la[:, None])[:, None, :]
    g[:, 0, 0] *= 1 - lh * la * rho
    g[:, 0, 1] *= 1 + lh * rho
    g[:, 1, 0] *= 1 + la * rho
    g[:, 1, 1] *= 1 - rho
    g /= g.sum((1, 2), keepdims=True)
    if draw_boost:
        idx = np.arange(len(k))
        g[:, idx, idx] *= 1 + draw_boost
        g /= g.sum((1, 2), keepdims=True)
    tri = np.tril(np.ones((len(k), len(k))), -1)
    p_home = (g * tri).sum((1, 2))
    p_draw = np.trace(g, axis1=1, axis2=2)
    likely = g.reshape(len(lh), -1).argmax(1)
    return np.stack([p_home, p_draw, 1 - p_home - p_draw], 1), np.divmod(likely, len(k))


def walk_forward(df, pre_elo, variants, start, refit_days=91, end=None):
    """Returns {variant name: DataFrame of per-match predictions} for matches from `start`."""
    pre_h, pre_a = pre_elo
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else df["date"].max() + pd.Timedelta(days=1)
    test = np.where((df["date"] >= start) & (df["date"] < end))[0]
    out = {}
    for name, settings in variants.items():
        probs = np.zeros((len(test), 3))
        lam = np.zeros((len(test), 2))
        likely = np.zeros((len(test), 2), int)
        cut = start
        while cut < end:
            nxt = cut + pd.Timedelta(days=refit_days)
            sel = np.where((df["date"].iloc[test] >= cut) & (df["date"].iloc[test] < nxt))[0]
            if len(sel):
                m = Model.fit(df, cutoff=cut, settings=settings, pre_elo=pre_elo, final_elo={})
                rows = test[sel]
                lh, la = rates_vec(m, df.iloc[rows], pre_h[rows], pre_a[rows])
                p, (i, j) = outcome_vec(lh, la, m.rho, m.draw_boost)
                probs[sel], lam[sel] = p, np.stack([lh, la], 1)
                likely[sel] = np.stack([i, j], 1)
            cut = nxt
        d = df.iloc[test]
        res = pd.DataFrame({
            "date": d["date"].to_numpy(), "competition": d["tournament"].to_numpy(),
            "home": d["home_team"].to_numpy(), "away": d["away_team"].to_numpy(),
            "p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2],
            "xg_home": lam[:, 0], "xg_away": lam[:, 1],
            "likely_score": [f"{a}-{b}" for a, b in likely],
            "score": [f"{a}-{b}" for a, b in zip(d["home_score"], d["away_score"])],
        })
        hs, as_ = d["home_score"].to_numpy(), d["away_score"].to_numpy()
        res["outcome"] = np.where(hs > as_, 0, np.where(hs == as_, 1, 2))
        res["pick"] = probs.argmax(1)
        res["correct"] = res["pick"] == res["outcome"]
        res["exact_score"] = res["likely_score"] == res["score"]
        out[name] = res
    return out


def metrics(res):
    p = res[["p_home", "p_draw", "p_away"]].to_numpy()
    s = score(p, res["outcome"].to_numpy())
    s["exact_score"] = float(res["exact_score"].mean())
    s["n"] = len(res)
    return s


def compare(results, title=""):
    if title:
        print(title)
    print(f"  {'variant':44s} {'n':>6s} {'right':>7s} {'exact':>6s} {'logloss':>8s} {'brier':>6s} {'RPS':>7s}")
    for name, res in results.items():
        s = metrics(res)
        print(f"  {name:44s} {s['n']:6d} {100 * s['accuracy']:6.1f}% {100 * s['exact_score']:5.1f}%"
              f" {s['log_loss']:8.4f} {s['brier']:6.4f} {s['rps']:7.4f}")


def calibration(res, bins=(0, .2, .35, .5, .65, .8, 1.0)):
    """Predicted vs actual frequency, for home win / draw / away win."""
    rows = []
    for k, label in enumerate(["home win", "draw", "away win"]):
        p = res[["p_home", "p_draw", "p_away"]].to_numpy()[:, k]
        hit = (res["outcome"] == k).to_numpy()
        cut = pd.cut(p, bins)
        for b, g in pd.DataFrame({"p": p, "hit": hit, "b": cut}).groupby("b", observed=True):
            rows.append((label, str(b), len(g), g["p"].mean(), g["hit"].mean()))
    return pd.DataFrame(rows, columns=["outcome", "bin", "n", "predicted", "actual"])
