"""Replay every game: predict each one using only earlier information, then score it.

python -m predictor evaluate          # internationals since 2000, clubs since 2017
python -m predictor evaluate --quick  # internationals since 2018, clubs since 2024

Writes predictor/reports/: report.md, report.json and one CSV per game set with every
prediction next to the real result.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clubs import CLUB_FULL_SETTINGS, CLUB_SETTINGS, club_elo, club_matches_with_features
from .data import DATA_DIR, load_results
from .elo import run_elo
from .evaluate import calibration, metrics, walk_forward
from .features import add_features
from .model import NATION_SETTINGS, Settings

REPORT_DIR = Path(__file__).parent / "reports"
ODDS_URL = ("https://raw.githubusercontent.com/AnishKhetani/premier-league-data/main/"
            "data/processed/results_with_odds.csv")
OLD, NEW = "before", "after"
MAJOR = ["FIFA World Cup", "UEFA Euro", "Copa América", "African Cup of Nations",
         "AFC Asian Cup", "Gold Cup"]


def _table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n"


def _summary_rows(groups, old, new):
    rows = []
    for label, mask in groups:
        if mask.sum() < 30:
            continue
        a, b = metrics(old[mask]), metrics(new[mask])
        rows.append([label, b["n"], f"{100 * b['accuracy']:.1f}%", f"{100 * b['exact_score']:.1f}%",
                     f"{a['log_loss']:.4f}", f"{b['log_loss']:.4f}",
                     f"{100 * (a['log_loss'] - b['log_loss']) / a['log_loss']:+.2f}%"])
    return rows


COLS = ["games", "n", "right result", "exact score", "log loss before", "log loss after", "improvement"]


def bookmaker_benchmark(club_new, club_old):
    path = DATA_DIR / "odds" / "pl_results_with_odds.csv"
    if not path.exists():
        import requests
        try:
            r = requests.get(ODDS_URL, timeout=120)
            r.raise_for_status()
        except Exception as e:
            return f"(Bookmaker odds not available: {e})\n", None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    from .backtest import score
    from .live_pl import _name_map
    odds = pd.read_csv(path, parse_dates=["date"])
    odds = odds[odds["date"] >= club_new["date"].min()]
    ours = sorted(set(club_new.loc[club_new["competition"] == "Premier League", "home"]))
    names = _name_map(set(odds["home_team"]) | set(odds["away_team"]), ours)
    odds = odds.assign(home=odds["home_team"].map(names), away=odds["away_team"].map(names))
    keep = ["date", "home", "away", "market_avg_1x2_home", "market_avg_1x2_draw", "market_avg_1x2_away",
            "market_avg_1x2_home_close", "market_avg_1x2_draw_close", "market_avg_1x2_away_close"]
    rows, out = [], {}
    for label, res in (("Model before", club_old), ("Model after", club_new)):
        e = res[res["competition"] == "Premier League"].merge(odds[keep], on=["date", "home", "away"])
        e = e.dropna(subset=keep[3:])
        s = score(e[["p_home", "p_draw", "p_away"]].to_numpy(), e["outcome"].to_numpy())
        rows.append([label, len(e), f"{100 * s['accuracy']:.1f}%", f"{s['log_loss']:.4f}"])
        out[label] = s
    for label, suf in (("Bookmakers, opening odds", ""), ("Bookmakers, closing odds", "_close")):
        q = 1 / e[[f"market_avg_1x2_home{suf}", f"market_avg_1x2_draw{suf}", f"market_avg_1x2_away{suf}"]].to_numpy()
        q /= q.sum(1, keepdims=True)
        s = score(q, e["outcome"].to_numpy())
        rows.append([label, len(e), f"{100 * s['accuracy']:.1f}%", f"{s['log_loss']:.4f}"])
        out[label] = s
    span = f"{e['date'].min().date()} to {e['date'].max().date()}"
    return (f"Premier League games {span} with average market odds:\n\n"
            + _table(rows, ["predictor", "games", "right result", "log loss"]), out)


def run_report(quick=False):
    REPORT_DIR.mkdir(exist_ok=True)
    md = ["# Every game, replayed\n",
          "Each game is predicted using only results and player data from before it "
          "(the model is refitted every few months as the replay moves forward). "
          "**Before** = Elo + Dixon-Coles goals model. **After** = adds squad market value, "
          "injuries, merged club names and (clubs) a draw adjustment. "
          "Log loss: lower is better; it rewards confident correct predictions and punishes "
          "confident wrong ones.\n"]
    summary = {}

    # ---- national teams
    start = "2018-01-01" if quick else "2000-01-01"
    print(f"Replaying internationals since {start}...")
    nat = add_features(load_results(), "nation")
    _, h, a = run_elo(nat)
    r = walk_forward(nat, (h, a), {OLD: Settings(), NEW: NATION_SETTINGS}, start, refit_days=182)
    old, new = r[OLD], r[NEW]
    new.to_csv(REPORT_DIR / "predictions_internationals.csv.gz", index=False, float_format="%.4f")
    yr = new["date"].dt.year
    groups = [("All internationals", np.ones(len(new), bool)),
              ("World Cup & continental finals", new["competition"].isin(MAJOR).to_numpy()),
              ("FIFA World Cup only", (new["competition"] == "FIFA World Cup").to_numpy()),
              ("Qualifiers", new["competition"].str.contains("qualification").to_numpy()),
              ("Friendlies", (new["competition"] == "Friendly").to_numpy())]
    md += [f"\n## National teams ({new['date'].min().date()} to {new['date'].max().date()})\n\n",
           _table(_summary_rows(groups, old, new), COLS)]
    md += ["\nBy year:\n\n", _table(_summary_rows(
        [(str(y), (yr == y).to_numpy()) for y in sorted(yr.unique())], old, new), COLS)]
    summary["internationals"] = {OLD: metrics(old), NEW: metrics(new)}

    # ---- clubs
    start = "2024-07-01" if quick else "2017-08-01"
    print(f"Replaying league games since {start}...")
    played, _ = club_matches_with_features()
    _, h, a = club_elo(played)
    r = walk_forward(played, (h, a), {OLD: CLUB_SETTINGS, NEW: CLUB_FULL_SETTINGS}, start, refit_days=60)
    cold, cnew = r[OLD], r[NEW]
    cnew.to_csv(REPORT_DIR / "predictions_clubs.csv.gz", index=False, float_format="%.4f")
    season = np.where(cnew["date"].dt.month >= 7, cnew["date"].dt.year, cnew["date"].dt.year - 1)
    groups = [("All league games", np.ones(len(cnew), bool))]
    groups += [(lg, (cnew["competition"] == lg).to_numpy()) for lg in sorted(cnew["competition"].unique())]
    md += [f"\n## Club leagues ({cnew['date'].min().date()} to {cnew['date'].max().date()})\n\n",
           _table(_summary_rows(groups, cold, cnew), COLS)]
    md += ["\nBy season (player data only exists from 2024-25, so earlier seasons barely change):\n\n",
           _table(_summary_rows([(f"{s}-{str(s + 1)[-2:]}", season == s) for s in sorted(set(season))],
                                cold, cnew), COLS)]
    summary["clubs"] = {OLD: metrics(cold), NEW: metrics(cnew)}

    # ---- calibration
    md += ["\n## Are the percentages honest? (all games, model after)\n\n"
           "When the model says 60%, does it happen about 60% of the time?\n\n"]
    cal = calibration(pd.concat([new, cnew]))
    md.append(_table([[c.outcome, c.bin, c.n, f"{100 * c.predicted:.1f}%", f"{100 * c.actual:.1f}%"]
                      for c in cal.itertuples()], ["outcome", "predicted range", "games", "predicted", "happened"]))

    # ---- bookmakers
    md.append("\n## Against the bookmakers\n\n")
    text, odds = bookmaker_benchmark(cnew, cold)
    md.append(text)
    summary["bookmakers"] = odds

    # ---- upsets
    allp = pd.concat([new.assign(kind="international"), cnew.assign(kind="club")])
    p_true = allp[["p_home", "p_draw", "p_away"]].to_numpy()[np.arange(len(allp)), allp["outcome"]]
    allp["p_actual"] = p_true
    recent = allp[allp["date"] >= allp["date"].max() - pd.Timedelta(days=730)]
    notable = (recent["kind"] == "club") | recent["competition"].isin(MAJOR) \
        | recent["competition"].str.contains("qualification|Nations League", regex=True)
    recent = recent[notable]
    md.append("\n## Biggest surprises of the last two years (leagues and competitive internationals)\n\n")
    md.append(_table([[d.date.date(), d.competition, f"{d.home} {d.score} {d.away}",
                       f"{100 * d.p_actual:.1f}%"] for d in recent.nsmallest(12, "p_actual").itertuples()],
                     ["date", "competition", "result", "model's chance of that result"]))

    (REPORT_DIR / "report.md").write_text("".join(md))
    (REPORT_DIR / "report.json").write_text(json.dumps(summary, indent=2, default=float))
    print("".join(md))
    print(f"\nWrote {REPORT_DIR}/report.md and per-game predictions.")
