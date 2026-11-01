"""Club football: Europe's top leagues from the open openfootball dataset.

Each league is rated as its own pool (there are no cross-league matches in the
data), so predictions are for matches inside one league.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import DATA_DIR
from .elo import run_elo, START
from .model import Model, Settings

CLUB_DIR = DATA_DIR / "clubs"
BASE_URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"
LEAGUES = {
    "en.1": "Premier League", "es.1": "La Liga", "de.1": "Bundesliga",
    "it.1": "Serie A", "fr.1": "Ligue 1", "pt.1": "Primeira Liga", "nl.1": "Eredivisie",
}
FIRST_SEASON = 2015
CURRENT_SEASON = 2026  # 2026-27
CLUB_SETTINGS = Settings(window_years=4, half_life_years=2.0, ridge=4.0)
MODEL_PATH = CLUB_DIR / "model.pkl"


def season_name(year):
    return f"{year}-{str(year + 1)[-2:]}"


def update_club_data(all_seasons=False):
    """Download league files. By default only the current season is refreshed."""
    CLUB_DIR.mkdir(parents=True, exist_ok=True)
    years = range(FIRST_SEASON, CURRENT_SEASON + 1)
    for y in years:
        for code in LEAGUES:
            path = CLUB_DIR / f"{season_name(y)}_{code}.json"
            if path.exists() and y != CURRENT_SEASON and not all_seasons:
                continue
            r = requests.get(BASE_URL.format(season=season_name(y), code=code), timeout=60)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            path.write_bytes(r.content)
    print("Club data updated.")


def _score(m):
    s = m.get("score")
    if isinstance(s, list) and len(s) == 2:  # some 0-0 results are stored as a plain list
        return s
    if isinstance(s, dict) and s.get("ft"):
        return s["ft"]
    return None


def load_club_matches():
    """Return (played matches, unplayed fixtures), both oldest first."""
    files = sorted(CLUB_DIR.glob("*.json")) if CLUB_DIR.exists() else []
    if not files:
        update_club_data()
        files = sorted(CLUB_DIR.glob("*.json"))
    rows = []
    for f in files:
        season, code = f.stem.split("_")
        for m in json.loads(f.read_text())["matches"]:
            sc = _score(m)
            rows.append({
                "date": m.get("date"), "time": m.get("time", ""),
                "home_team": m["team1"], "away_team": m["team2"],
                "home_score": sc[0] if sc else None, "away_score": sc[1] if sc else None,
                "tournament": LEAGUES[code], "league": code, "season": season, "neutral": False,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played[["home_score", "away_score"]] = played[["home_score", "away_score"]].astype(int)
    fixtures = df[df["home_score"].isna()].reset_index(drop=True)
    return played.reset_index(drop=True), fixtures


def club_elo(df):
    """Elo where a club promoted into a league starts at the average rating of
    the clubs that went down from it the season before."""
    members = df.groupby(["league", "season"])["home_team"].agg(set).to_dict()
    reset_at, start_of = {}, {}
    for (lg, season), teams in members.items():
        prev = season_name(int(season[:4]) - 1)
        if (lg, prev) not in members:
            continue
        newcomers = teams - members[(lg, prev)]
        relegated = members[(lg, prev)] - teams
        rows = df.index[(df["league"] == lg) & (df["season"] == season)]
        for t in newcomers:
            first = rows[(df.loc[rows, "home_team"] == t) | (df.loc[rows, "away_team"] == t)][0]
            reset_at.setdefault(first, set()).add(t)
            start_of[(first, t)] = relegated

    def entry(team, i, ratings):
        down = [ratings[t] for t in start_of.get((i, team), ()) if t in ratings]
        return float(np.mean(down)) if down else START - 100  # unknown newcomer: below average

    return run_elo(df, new_team_rating=entry, reset_at=reset_at)


def get_club_model(refit=False):
    played, _ = load_club_matches()
    newest = max(p.stat().st_mtime for p in CLUB_DIR.glob("*.json"))
    if not refit and MODEL_PATH.exists() and MODEL_PATH.stat().st_mtime >= newest:
        return Model.load(MODEL_PATH)
    elo, pre_h, pre_a = club_elo(played)
    m = Model.fit(played, settings=CLUB_SETTINGS, pre_elo=(pre_h, pre_a), final_elo=elo)
    m.save(MODEL_PATH)
    return m
