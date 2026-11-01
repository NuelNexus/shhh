"""Extra match factors on top of Elo and the goals model.

squad_diff  difference in squad market value (log), from Transfermarkt
avail_diff  difference in how much of that value is fit (injuries)
rest_diff   days since the previous league game (clubs; tested but not used: the data
            has no cup or European games, so it misreads rest for busy teams)

All of them only use information available before kick-off.
"""
import numpy as np

from .players.features import DIR as PLAYER_DIR

NATION_FEATURES = ("squad_diff", "avail_diff")
CLUB_FEATURES = ("squad_diff", "avail_diff")

_tables = None


def player_tables():
    global _tables
    if _tables is None and (PLAYER_DIR / "players.csv.gz").exists():
        from .players.features import PlayerTables
        _tables = PlayerTables()
    return _tables


def rest_days(df):
    last, rh, ra = {}, np.full(len(df), np.nan), np.full(len(df), np.nan)
    for i, (d, h, a) in enumerate(zip(df["date"], df["home_team"], df["away_team"])):
        if h in last:
            rh[i] = (d - last[h]).days
        if a in last:
            ra[i] = (d - last[a]).days
        last[h] = last[a] = d
    clip = lambda x: np.clip(np.nan_to_num(x, nan=7), 3, 8)
    return clip(rh), clip(ra)


def add_features(df, kind):
    """Return df with the factor columns added. kind: 'nation' or 'club'."""
    from .players.features import match_features
    df = df.copy()
    tables = player_tables()
    if tables is not None:
        if kind == "club":
            f = match_features(df, tables, "home_tm_id", "away_tm_id", "club")
        else:
            f = match_features(df, tables, "home_team", "away_team", "nation")
        df["squad_diff"], df["avail_diff"] = f["squad_diff"], f["avail_diff"]
    else:
        df["squad_diff"] = df["avail_diff"] = np.nan
    if kind == "club":
        rh, ra = rest_days(df)
        df["rest_diff"] = rh - ra
    return df
