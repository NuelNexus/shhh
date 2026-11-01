"""Turn a fitted model into a full match prediction."""
import difflib

import numpy as np
from scipy.stats import poisson


def resolve_team(name, model):
    teams = sorted(model.elo)
    lookup = {t.lower(): t for t in teams}
    if name.lower() in lookup:
        return lookup[name.lower()]
    close = difflib.get_close_matches(name, teams, n=5, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise SystemExit(f"Unknown team '{name}'.{hint}")


def predict_match(model, team_a, team_b, neutral=True, knockout=False):
    """team_a is the home side when neutral=False."""
    a, b = resolve_team(team_a, model), resolve_team(team_b, model)
    la, lb = model.rates(a, b, neutral)
    g = model.score_grid(la, lb)
    p_a, p_d, p_b = np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()

    flat = [((i, j), g[i, j]) for i in range(g.shape[0]) for j in range(g.shape[1])]
    top = sorted(flat, key=lambda x: -x[1])[:5]
    totals = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))

    out = {
        "team_a": a, "team_b": b, "neutral": neutral,
        "elo_a": model.elo.get(a), "elo_b": model.elo.get(b),
        "xg_a": la, "xg_b": lb,
        "win_a": p_a, "draw": p_d, "win_b": p_b,
        "top_scores": top,
        "over_2_5": g[totals > 2.5].sum(),
        "btts": g[1:, 1:].sum(),
    }
    if knockout:
        # 30 minutes of extra time = one third of the goal rate, then a 50/50 shootout.
        et = np.outer(poisson.pmf(np.arange(8), la / 3), poisson.pmf(np.arange(8), lb / 3))
        et /= et.sum()
        et_a, et_d = np.tril(et, -1).sum(), np.trace(et)
        out["advance_a"] = p_a + p_d * (et_a + et_d * 0.5)
        out["advance_b"] = 1 - out["advance_a"]
    return out


def format_prediction(p):
    a, b = p["team_a"], p["team_b"]
    pct = lambda x: f"{100 * x:5.1f}%"
    w = max(len(a), len(b)) + 5
    venue = "neutral venue" if p["neutral"] else f"{a} at home"
    lines = [
        f"\n{a} vs {b}  ({venue})",
        f"Elo: {a} {p['elo_a']:.0f}  |  {b} {p['elo_b']:.0f}",
        "",
        f"  {a + ' win':{w}s} {pct(p['win_a'])}",
        f"  {'Draw':{w}s} {pct(p['draw'])}",
        f"  {b + ' win':{w}s} {pct(p['win_b'])}",
        "",
        f"Expected goals: {a} {p['xg_a']:.2f} - {p['xg_b']:.2f} {b}",
        "Most likely scores:",
    ]
    for (i, j), pr in p["top_scores"]:
        lines.append(f"  {a} {i}-{j} {b}   {pct(pr)}")
    lines += [
        "",
        f"Over 2.5 goals: {pct(p['over_2_5'])}   Both teams score: {pct(p['btts'])}",
    ]
    if "advance_a" in p:
        lines.append(f"To advance (incl. extra time & penalties): "
                     f"{a} {pct(p['advance_a'])}  |  {b} {pct(p['advance_b'])}")
    fav = max([(p["win_a"], f"{a} to win"), (p["draw"], "Draw"), (p["win_b"], f"{b} to win")])
    (si, sj), _ = p["top_scores"][0]
    lines.append(f"\nPick: {fav[1]} ({pct(fav[0]).strip()}), most likely score {si}-{sj}")
    return "\n".join(lines)
