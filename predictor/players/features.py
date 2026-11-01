"""Player-based match factors, using only what was known before kick-off.

For each team on a match date:
  squad_value  log of the combined market value of its N most valuable players
  available    log of (value of the best N fit players / value of the best N players),
               so 0 means nobody important is injured and -0.3 means about a quarter
               of the squad's value is missing
Club squads come from Transfermarkt's current-squad snapshot (Sept 2025), so they are
only trusted from CLUB_START onward. National teams use each nation's most valuable
players at the time (no knowledge of the actual call-ups).
"""
import numpy as np
import pandas as pd

from ..data import DATA_DIR

DIR = DATA_DIR / "players"
CLUB_START = pd.Timestamp("2024-07-01")
NATION_START = pd.Timestamp("2012-01-01")  # market values are too sparse before ~2010
CLUB_N, NATION_N = 18, 23
STALE_DAYS = 550  # a valuation older than this means the player has dropped out
MIN_PLAYERS = 11
INJURY_FRESH_DAYS = 21  # past the end of the injury data, availability is unknown


class PlayerTables:
    def __init__(self):
        self.players = pd.read_csv(DIR / "players.csv.gz")
        self.players["joined"] = pd.to_datetime(self.players["joined"], errors="coerce")
        vals = pd.read_csv(DIR / "values.csv.gz", parse_dates=["date"])
        inj = pd.read_csv(DIR / "injuries.csv.gz")
        inj["start"] = pd.to_datetime(inj["start"], errors="coerce")
        inj["end"] = pd.to_datetime(inj["end"], errors="coerce")
        inj = inj.dropna(subset=["start", "end"])
        self.vals = {pid: (g["date"].to_numpy("datetime64[D]"), g["value"].to_numpy(float))
                     for pid, g in vals.groupby("player_id")}
        self.inj = {pid: (g["start"].to_numpy("datetime64[D]"), g["end"].to_numpy("datetime64[D]"))
                    for pid, g in inj.groupby("player_id")}
        self.injuries_until = inj["start"].max()
        self.values_until = vals["date"].max()
        self.names = dict(zip(self.players["player_id"], self.players["player_name"]))
        clubbed = self.players.dropna(subset=["current_club_id"])
        self.club_squads = {int(c): g for c, g in clubbed.groupby("current_club_id")}
        self.nation_pools = {n: g for n, g in self.players.groupby("nation")}

    def _matrices(self, pids, joined, dates):
        """Value and injury matrices, shape (len(dates), len(pids))."""
        d = np.asarray(dates, dtype="datetime64[D]")
        value = np.zeros((len(d), len(pids)))
        hurt = np.zeros((len(d), len(pids)), dtype=bool)
        for j, pid in enumerate(pids):
            if pid in self.vals:
                vd, vv = self.vals[pid]
                k = np.searchsorted(vd, d, side="left") - 1  # last valuation strictly before
                ok = k >= 0
                kk = np.clip(k, 0, None)
                fresh = ok & ((d - vd[kk]).astype(int) <= STALE_DAYS)
                value[:, j] = np.where(fresh, vv[kk], 0.0)
            if joined is not None and not pd.isna(joined[j]):
                value[d < np.datetime64(joined[j], "D"), j] = 0.0
            if pid in self.inj:
                s, e = self.inj[pid]
                # injured before match day and not yet back
                hurt[:, j] = ((s[None, :] < d[:, None]) & (e[None, :] >= d[:, None])).any(1)
        return value, hurt

    def _stale(self, dates):
        limit = np.datetime64(self.injuries_until + pd.Timedelta(days=INJURY_FRESH_DAYS), "D")
        return np.asarray(dates, dtype="datetime64[D]") > limit

    @staticmethod
    def _summarise(value, hurt, n):
        top_all = -np.sort(-value, axis=1)[:, :n]
        top_fit = -np.sort(-np.where(hurt, 0.0, value), axis=1)[:, :n]
        total = top_all.sum(1)
        count = (top_all > 0).sum(1)
        squad = np.log1p(total / 1e6)
        avail = np.log(np.maximum(top_fit.sum(1), 1.0) / np.maximum(total, 1.0))
        bad = count < MIN_PLAYERS
        return np.where(bad, np.nan, squad), np.where(bad, np.nan, avail)

    def club_features(self, tm_id, dates):
        g = self.club_squads.get(int(tm_id)) if not pd.isna(tm_id) else None
        if g is None:
            nan = np.full(len(dates), np.nan)
            return nan, nan
        value, hurt = self._matrices(g["player_id"].to_numpy(), g["joined"].to_numpy(), dates)
        squad, avail = self._summarise(value, hurt, CLUB_N)
        early = np.asarray(dates, dtype="datetime64[D]") < np.datetime64(CLUB_START, "D")
        return np.where(early, np.nan, squad), np.where(early | self._stale(dates), np.nan, avail)

    def nation_features(self, nation, dates):
        g = self.nation_pools.get(nation)
        if g is None:
            nan = np.full(len(dates), np.nan)
            return nan, nan
        value, hurt = self._matrices(g["player_id"].to_numpy(), None, dates)
        squad, avail = self._summarise(value, hurt, NATION_N)
        early = np.asarray(dates, dtype="datetime64[D]") < np.datetime64(NATION_START, "D")
        return np.where(early, np.nan, squad), np.where(early | self._stale(dates), np.nan, avail)

    def missing_players(self, pids_or_team, date, kind="club", top=8):
        """Who is injured on `date` among the team's most valuable players (for display)."""
        g = (self.club_squads.get(int(pids_or_team)) if kind == "club"
             else self.nation_pools.get(pids_or_team))
        if g is None:
            return []
        n = CLUB_N if kind == "club" else NATION_N
        value, hurt = self._matrices(g["player_id"].to_numpy(),
                                     g["joined"].to_numpy() if kind == "club" else None, [date])
        order = np.argsort(-value[0])[:n]
        return [(self.names.get(g["player_id"].iat[j], "?"), value[0, j]) for j in order
                if hurt[0, j] and value[0, j] > 0][:top]


def match_features(df, tables, key_home, key_away, kind):
    """Add squad/availability diffs (home minus away) to a match DataFrame."""
    out = {f"{side}_{f}": np.full(len(df), np.nan)
           for side in ("home", "away") for f in ("squad", "avail")}
    fn = tables.club_features if kind == "club" else tables.nation_features
    for side, col in (("home", key_home), ("away", key_away)):
        for team, idx in df.groupby(col).indices.items():
            squad, avail = fn(team, df["date"].iloc[idx].to_numpy())
            out[f"{side}_squad"][idx] = squad
            out[f"{side}_avail"][idx] = avail
    feats = pd.DataFrame(out, index=df.index)
    feats["squad_diff"] = feats["home_squad"] - feats["away_squad"]
    feats["avail_diff"] = feats["home_avail"] - feats["away_avail"]
    return feats
