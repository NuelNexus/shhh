"""Live Premier League data from the open pl-catalogue project
(https://github.com/oladeeayo/pl-catalogue, refreshed after each gameweek):

- results newer than the main league data
- each player's current status from the official Fantasy Premier League feed
  (injured / doubtful with % chance / suspended) and Transfermarkt value
"""
import difflib

import numpy as np
import pandas as pd
import requests

from .data import DATA_DIR

SEASON = "2026-27"
URL = "https://raw.githubusercontent.com/oladeeayo/pl-catalogue/main/data/{season}/{name}"
LIVE_DIR = DATA_DIR / "live" / "pl"
ALIASES = {
    "Spurs": "Tottenham Hotspur FC", "Man Utd": "Manchester United FC",
    "Man City": "Manchester City FC", "Nott'm Forest": "Nottingham Forest FC",
    "Leeds": "Leeds United FC", "Newcastle": "Newcastle United FC",
    "Brighton": "Brighton & Hove Albion FC", "Bournemouth": "AFC Bournemouth",
    "Wolves": "Wolverhampton Wanderers FC", "West Ham": "West Ham United FC",
}
SQUAD_N = 18


def update_live():
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("fixtures.csv", "players.csv", "manifest.json"):
        r = requests.get(URL.format(season=SEASON, name=name), timeout=60)
        r.raise_for_status()
        (LIVE_DIR / name).write_bytes(r.content)
    print("Live Premier League data updated.")


def _name_map(names, league_names):
    out = {}
    for n in names:
        out[n] = ALIASES.get(n) or difflib.get_close_matches(n + " FC", league_names, 1, 0.0)[0]
    return out


def add_live_results(played, fixtures):
    """Fill in Premier League results the main data source doesn't have yet."""
    path = LIVE_DIR / "fixtures.csv"
    if not path.exists():
        return played, fixtures
    live = pd.read_csv(path)
    live = live[live["finished"] == 1]
    epl = pd.concat([played, fixtures])
    epl = epl[(epl["league"] == "en.1") & (epl["season"] == SEASON)]
    names = _name_map(set(live["home_team"]) | set(live["away_team"]), sorted(set(epl["home_team"])))
    live["home"], live["away"] = live["home_team"].map(names), live["away_team"].map(names)
    scores = {(h, a): (int(x), int(y)) for h, a, x, y in
              zip(live["home"], live["away"], live["home_score"], live["away_score"])}
    is_fix = (fixtures["league"] == "en.1") & (fixtures["season"] == SEASON)
    keys = list(zip(fixtures["home_team"], fixtures["away_team"]))
    now_played = np.array([m and k in scores for m, k in zip(is_fix, keys)], dtype=bool)
    newly = fixtures[now_played].copy()
    newly["home_score"] = [scores[(h, a)][0] for h, a in zip(newly["home_team"], newly["away_team"])]
    newly["away_score"] = [scores[(h, a)][1] for h, a in zip(newly["home_team"], newly["away_team"])]
    played = pd.concat([played, newly]).sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    return played, fixtures[~now_played].reset_index(drop=True)


def team_status(league_names):
    """Current squad value, share fit, and who's out, for each Premier League club."""
    path = LIVE_DIR / "players.csv"
    if not path.exists():
        return {}
    p = pd.read_csv(path)
    p = p[(p["status"] != "u") & (p["market_value_eur"] > 0)].copy()  # 'u' = left the club
    chance = p["chance_of_playing_next_round"].fillna(100) / 100
    p["fit_value"] = p["market_value_eur"] * np.where(p["status"] == "a", 1.0, chance)
    names = _name_map(set(p["team"]), league_names)
    out = {}
    for team, g in p.groupby("team"):
        top = np.sort(g["market_value_eur"].to_numpy())[::-1][:SQUAD_N]
        fit = np.sort(g["fit_value"].to_numpy())[::-1][:SQUAD_N]
        total = top.sum()
        out_list = g[g["status"] != "a"].sort_values("market_value_eur", ascending=False)
        out[names[team]] = {
            "squad": np.log1p(total / 1e6),
            "avail": np.log(max(fit.sum(), 1.0) / max(total, 1.0)),
            "missing": [(r.web_name, r.news) for r in out_list.itertuples()][:8],
        }
    return out


def as_of():
    path = LIVE_DIR / "manifest.json"
    if not path.exists():
        return None
    import json
    return json.loads(path.read_text()).get("exported_at", "")[:10]
