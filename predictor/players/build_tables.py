"""One-off script: shrink the Transfermarkt data lake to the players this project needs.

Run:  python -m predictor.players.build_tables /path/to/salimt/football-datasets
Source: https://github.com/salimt/football-datasets (Transfermarkt scrape).
Writes predictor/data/players/{players,values,injuries}.csv.gz
"""
import sys
from pathlib import Path

import pandas as pd

from ..data import DATA_DIR

OUT = DATA_DIR / "players"
NATION_FIX = {
    "Türkiye": "Turkey", "Korea, South": "South Korea", "Korea, North": "North Korea",
    "Ireland": "Republic of Ireland", "Cote d'Ivoire": "Ivory Coast",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina", "The Gambia": "Gambia",
    "Curacao": "Curaçao", "St. Kitts & Nevis": "Saint Kitts and Nevis",
    "Saint-Martin": "Saint Martin", "Hongkong": "Hong Kong", "St. Lucia": "Saint Lucia",
    "Neukaledonien": "New Caledonia", "St. Vincent & Grenadinen": "Saint Vincent and the Grenadines",
    "Southern Sudan": "South Sudan", "American Virgin Islands": "United States Virgin Islands",
    "Macao": "Macau", "Brunei Darussalam": "Brunei", "Chinese Taipei": "Taiwan",
    "Turks- and Caicosinseln": "Turks and Caicos Islands",
}
POOL_PER_YEAR = 50  # keep each nation's 50 most valuable players in every year


def main(root):
    tm = Path(root) / "datalake/transfermarkt"
    OUT.mkdir(parents=True, exist_ok=True)
    prof = pd.read_csv(tm / "player_profiles/player_profiles.csv", low_memory=False,
                       usecols=["player_id", "player_name", "citizenship", "current_club_id",
                                "joined", "main_position"])
    first = prof["citizenship"].fillna("").str.split("  ").str[0].str.strip()
    prof["nation"] = first.replace(NATION_FIX)
    prof["player_name"] = prof["player_name"].str.replace(r"\s*\(\d+\)$", "", regex=True)

    vals = pd.read_csv(tm / "player_market_value/player_market_value.csv")
    vals["date"] = pd.to_datetime(vals["date_unix"], errors="coerce")
    vals = vals.dropna(subset=["date"])
    vals = vals[(vals["value"] > 0) & (vals["date"] >= "2004-01-01")]

    # National pools: each nation's most valuable players, year by year.
    yearly = vals.assign(year=vals["date"].dt.year).groupby(["player_id", "year"])["value"].max()
    yearly = yearly.reset_index().merge(prof[["player_id", "nation"]], on="player_id")
    yearly["rank"] = yearly.groupby(["nation", "year"])["value"].rank(ascending=False, method="first")
    pool = set(yearly.loc[yearly["rank"] <= POOL_PER_YEAR, "player_id"])

    club_ids = pd.read_csv(DATA_DIR / "clubs/club_ids.csv")
    at_club = set(prof.loc[prof["current_club_id"].isin(club_ids["tm_id"]), "player_id"])
    keep = pool | at_club

    players = prof[prof["player_id"].isin(keep)].copy()
    players.loc[~players["player_id"].isin(at_club), ["current_club_id", "joined"]] = None
    players = players[["player_id", "player_name", "nation", "main_position", "current_club_id", "joined"]]
    players.to_csv(OUT / "players.csv.gz", index=False)

    v = vals[vals["player_id"].isin(keep)][["player_id", "date", "value"]]
    v.sort_values(["player_id", "date"]).to_csv(OUT / "values.csv.gz", index=False)

    inj = pd.read_csv(tm / "player_injuries/player_injuries_updated_20260929.csv")
    inj = inj[inj["player_id"].isin(keep)]
    inj = inj.rename(columns={"from_date": "start", "end_date": "end", "injury_reason": "reason"})
    inj[["player_id", "start", "end", "reason", "games_missed"]].to_csv(OUT / "injuries.csv.gz", index=False)

    print(f"players: {len(players)} ({len(at_club)} at league clubs, {len(pool)} in national pools)")
    print(f"values: {len(v)} rows, injuries: {len(inj)} rows")
    for f in OUT.glob("*.gz"):
        print(f.name, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1])
