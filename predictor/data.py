"""Load and refresh the international match dataset."""
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent / "data"
BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master/"
FILES = ["results.csv", "shootouts.csv"]


def update_data():
    """Download the latest match results."""
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        r = requests.get(BASE_URL + name, timeout=60)
        r.raise_for_status()
        (DATA_DIR / name).write_bytes(r.content)
        print(f"Downloaded {name} ({len(r.content) // 1024} KB)")


def load_results():
    """Return played matches, oldest first."""
    path = DATA_DIR / "results.csv"
    if not path.exists():
        update_data()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])  # skip unplayed fixtures
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df.sort_values("date", kind="stable").reset_index(drop=True)


# Competition importance, used for Elo K-factor and model weights.
CONTINENTAL_FINALS = (
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "Gold Cup", "CONCACAF Championship", "Oceania Nations Cup",
    "Confederations Cup", "CONMEBOL–UEFA Cup of Champions",
)


def competition_tier(tournament: str) -> str:
    t = tournament or ""
    if t == "FIFA World Cup":
        return "world_cup"
    if "qualification" in t:
        return "qualifier"
    if t in CONTINENTAL_FINALS:
        return "continental"
    if "Nations League" in t:
        return "nations_league"
    if t == "Friendly":
        return "friendly"
    return "other"


ELO_K = {
    "world_cup": 60, "continental": 50, "qualifier": 40,
    "nations_league": 40, "other": 30, "friendly": 20,
}
FIT_WEIGHT = {
    "world_cup": 1.0, "continental": 1.0, "qualifier": 1.0,
    "nations_league": 1.0, "other": 0.8, "friendly": 0.6,
}
