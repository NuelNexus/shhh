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
CLUB_SETTINGS = Settings(window_years=4, half_life_years=2.0, ridge=4.0)  # before player factors
CLUB_FULL_SETTINGS = Settings(window_years=4, half_life_years=2.0, ridge=4.0, fit_draws=True,
                              features=("squad_diff", "avail_diff"))
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


def load_club_matches(canonical=True):
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
    if canonical:
        df = canonicalise(df)
    df = df.sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played[["home_score", "away_score"]] = played[["home_score", "away_score"]].astype(int)
    fixtures = df[df["home_score"].isna()].reset_index(drop=True)
    return played.reset_index(drop=True), fixtures


def canonicalise(df):
    """The source spells some clubs differently across seasons ("Inter" vs
    "FC Internazionale Milano"). Give each club one name (its latest spelling)
    and attach its Transfermarkt id."""
    ids_path = CLUB_DIR / "club_ids.csv"
    if not ids_path.exists():
        df["tm_id"] = np.nan
        return df
    ids = pd.read_csv(ids_path)
    tm = dict(zip(ids["name"], ids["tm_id"]))
    latest = (df.assign(tm_id=df["home_team"].map(tm)).sort_values("date")
              .groupby("tm_id")["home_team"].last())
    canon = {name: latest.get(t, name) for name, t in tm.items()}
    for col in ("home_team", "away_team"):
        df[col] = df[col].map(lambda n: canon.get(n, n))
    df["home_tm_id"] = df["home_team"].map(tm)
    df["away_tm_id"] = df["away_team"].map(tm)
    return df


def club_elo(df, k_scale=1.0, home_bonus=100.0):
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

    return run_elo(df, new_team_rating=entry, reset_at=reset_at, k_scale=k_scale, home_bonus=home_bonus)


def club_matches_with_features():
    """All league games (played and scheduled) with factor columns, oldest first."""
    from .features import add_features
    from .live_pl import add_live_results
    played, fixtures = load_club_matches()
    played, fixtures = add_live_results(played, fixtures)
    allm = pd.concat([played, fixtures], ignore_index=True).sort_values(["date", "time"], kind="stable")
    allm = add_features(allm.reset_index(drop=True), "club")
    done = allm["home_score"].notna()
    played = allm[done].reset_index(drop=True)
    played[["home_score", "away_score"]] = played[["home_score", "away_score"]].astype(int)
    return played, allm[~done].reset_index(drop=True)


def get_club_model(refit=False):
    sources = list(CLUB_DIR.glob("*.json")) + list((DATA_DIR / "live").glob("**/*.csv"))
    newest = max(p.stat().st_mtime for p in sources)
    if not refit and MODEL_PATH.exists() and MODEL_PATH.stat().st_mtime >= newest:
        return Model.load(MODEL_PATH)
    played, _ = club_matches_with_features()
    elo, pre_h, pre_a = club_elo(played)
    m = Model.fit(played, settings=CLUB_FULL_SETTINGS, pre_elo=(pre_h, pre_a), final_elo=elo)
    m.save(MODEL_PATH)
    return m


def resolve_club(name, model):
    """Find a club by loose name ('real madrid', 'racing santander', 'spurs')."""
    import difflib
    from .live_pl import ALIASES
    name = ALIASES.get(name, name)
    teams = sorted(model.elo)
    low = {t.lower(): t for t in teams}
    if name.lower() in low:
        return low[name.lower()]
    hits = [t for t in teams if all(w in t.lower() for w in name.lower().split())]
    if len(hits) == 1:
        return hits[0]
    close = difflib.get_close_matches(name, teams, n=5, cutoff=0.4)
    if hits:
        close = hits[:5]
    raise SystemExit(f"Unknown club '{name}'." + (f" Did you mean: {', '.join(close)}?" if close else ""))


def club_match_features(home, away, date, played=None):
    """Factor values for an upcoming club game, plus who's missing (if known)."""
    from .features import player_tables
    from .live_pl import team_status
    date = pd.Timestamp(date)
    if played is None:
        played, _ = load_club_matches()
    tm = dict(zip(played["home_team"], played["home_tm_id"]))
    league = played.loc[played["home_team"] == home, "league"]
    league = league.iloc[-1] if len(league) else None
    feats, missing, source = {}, {home: [], away: []}, None
    tables = player_tables()
    if tables is not None:
        sh, ah = tables.club_features(tm.get(home), [date])
        sa, aa = tables.club_features(tm.get(away), [date])
        feats = {"squad_diff": float(sh[0] - sa[0]), "avail_diff": float(ah[0] - aa[0])}
        source = f"Transfermarkt (squads Sep 2025, injuries to {tables.injuries_until.date()})"
    if league == "en.1":
        live = team_status(sorted(set(played.loc[played["league"] == "en.1", "home_team"])))
        if home in live and away in live:
            feats["squad_diff"] = live[home]["squad"] - live[away]["squad"]
            feats["avail_diff"] = live[home]["avail"] - live[away]["avail"]
            missing = {home: live[home]["missing"], away: live[away]["missing"]}
            from .live_pl import as_of
            source = f"Fantasy Premier League injury news ({as_of()})"
    last = {}
    for t in (home, away):
        g = played[((played["home_team"] == t) | (played["away_team"] == t)) & (played["date"] < date)]
        last[t] = (date - g["date"].max()).days if len(g) else 7
    feats["rest_diff"] = float(np.clip(last[home], 3, 8) - np.clip(last[away], 3, 8))
    return feats, missing, source, LEAGUES.get(league, league)
