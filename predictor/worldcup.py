"""Monte Carlo World Cup simulator.

Plays the whole tournament thousands of times using the match model:
group stage (with real scorelines, so goal difference counts), then knockouts
(with extra time and penalties). Supports the 32-team format (8 groups, top 2
go through) and the 48-team format (12 groups, top 2 + 8 best third-placed
teams go to a round of 32).

Simplification: in the 48-team format the official round-of-32 pairing table is
replaced by seeding the 32 qualifiers (group winners first) so the best teams
are kept apart until late, as the real bracket broadly does.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .model import Model

STAGES = ["R32", "R16", "QF", "SF", "Final", "Champion"]
HOSTS = {2026: ["United States", "Mexico", "Canada"], 2022: ["Qatar"], 2018: ["Russia"],
         2014: ["Brazil"], 2030: ["Spain", "Portugal", "Morocco"]}


def bracket_order(n):
    """Seed order so seed 1 and 2 can only meet in the final (1 v n, ...)."""
    order = [1]
    while len(order) < n:
        m = len(order) * 2 + 1
        order = [x for s in order for x in (s, m - s)]
    return order


class Simulator:
    def __init__(self, model: Model, hosts=(), rng=None):
        self.m = model
        self.hosts = set(hosts)
        self.rng = rng or np.random.default_rng()
        self._adv = {}

    def _neutral(self, a, b):
        return not (a in self.hosts and b not in self.hosts)

    def _grid(self, a, b):
        if b in self.hosts and a not in self.hosts:  # host listed second: flip
            return self._grid(b, a).T
        return self.m.score_grid(*self.m.rates(a, b, self._neutral(a, b)))

    def advance_prob(self, a, b):
        key = (a, b)
        if key not in self._adv:
            g = self._grid(a, b)
            pa, pd_ = np.tril(g, -1).sum(), np.trace(g)
            if b in self.hosts and a not in self.hosts:
                lb, la = self.m.rates(b, a, False)
            else:
                la, lb = self.m.rates(a, b, self._neutral(a, b))
            et = np.outer(poisson.pmf(np.arange(8), la / 3), poisson.pmf(np.arange(8), lb / 3))
            et /= et.sum()
            self._adv[key] = pa + pd_ * (np.tril(et, -1).sum() + np.trace(et) * 0.5)
        return self._adv[key]

    def play_groups(self, groups, n):
        """Returns per group an (n, 4) array of team indices in finishing order,
        plus (n,) ranking keys for the third-placed team."""
        finish, third_keys = [], []
        for teams in groups:
            k = len(teams)
            pts, gd, gf = np.zeros((n, k)), np.zeros((n, k)), np.zeros((n, k))
            for i, j in combinations(range(k), 2):
                g = self._grid(teams[i], teams[j])
                flat = self.rng.choice(g.size, size=n, p=g.ravel())
                x, y = np.divmod(flat, g.shape[1])
                pts[:, i] += np.where(x > y, 3, np.where(x == y, 1, 0))
                pts[:, j] += np.where(y > x, 3, np.where(x == y, 1, 0))
                gd[:, i] += x - y
                gd[:, j] += y - x
                gf[:, i] += x
                gf[:, j] += y
            key = pts * 1e6 + (gd + 100) * 1e3 + gf + self.rng.random((n, k))  # random = lots drawn
            order = np.argsort(-key, axis=1)
            finish.append(order)
            third_keys.append(np.take_along_axis(key, order[:, 2:3], 1)[:, 0])
        return finish, np.stack(third_keys, 1)

    def run(self, groups, n=10000):
        """groups: list of lists of team names. Returns a DataFrame of stage probabilities."""
        groups = [list(g) for g in groups]
        teams = [t for g in groups for t in g]
        reach = {t: np.zeros(len(STAGES)) for t in teams}
        finish, third_keys = self.play_groups(groups, n)
        n_groups = len(groups)
        big = n_groups == 12
        if n_groups not in (8, 12):
            raise ValueError("Need 8 groups (32 teams) or 12 groups (48 teams).")

        for s in range(n):
            if big:
                # seeds: 12 winners, 12 runners-up, 8 best thirds; ranked by group record
                ranked = []
                for pos in range(2):
                    ranked += [groups[gi][finish[gi][s, pos]] for gi in range(12)]
                best3 = np.argsort(-third_keys[s])[:8]
                ranked += [groups[gi][finish[gi][s, 2]] for gi in best3]
                # keep group winners on top, shuffle within each tier
                tiers = [ranked[:12], ranked[12:24], ranked[24:]]
                seeds = []
                for t in tiers:
                    seeds += list(self.rng.permutation(t))
                field = [seeds[k - 1] for k in bracket_order(32)]
                stage0 = 0
            else:
                w = [groups[gi][finish[gi][s, 0]] for gi in range(8)]
                r = [groups[gi][finish[gi][s, 1]] for gi in range(8)]
                # classic bracket: A1-B2, C1-D2, E1-F2, G1-H2, B1-A2, D1-C2, F1-E2, H1-G2
                field = [w[0], r[1], w[2], r[3], w[4], r[5], w[6], r[7],
                         w[1], r[0], w[3], r[2], w[5], r[4], w[7], r[6]]
                stage0 = 1
            stage = stage0
            while len(field) > 1:
                for t in field:
                    reach[t][stage] += 1
                nxt = []
                for a, b in zip(field[::2], field[1::2]):
                    nxt.append(a if self.rng.random() < self.advance_prob(a, b) else b)
                field = nxt
                stage += 1
            reach[field[0]][stage] += 1

        rows = []
        for gi, g in enumerate(groups):
            for t in g:
                r = reach[t] / n
                rows.append({"team": t, "group": chr(65 + gi), "elo": self.m.elo.get(t, np.nan),
                             **{st: r[k] for k, st in enumerate(STAGES)}})
        df = pd.DataFrame(rows)
        if not big:
            df = df.drop(columns="R32")
        return df.sort_values(["Champion", "Final", "SF"], ascending=False).reset_index(drop=True)


def groups_from_results(df, year):
    """Rebuild a past World Cup's groups from who played whom in the group stage."""
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == year)]
    n_group_games = 72 if len(wc) > 64 else 48
    gs = wc.iloc[:n_group_games]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    for h, a in zip(gs["home_team"], gs["away_team"]):
        parent[find(h)] = find(a)
    groups = {}
    for t in pd.unique(gs[["home_team", "away_team"]].to_numpy().ravel()):
        groups.setdefault(find(t), []).append(t)
    return list(groups.values()), wc


def actual_finish(wc, groups, shootouts=None):
    """Furthest stage each team reached in a real tournament."""
    n_group_games = 72 if len(wc) > 64 else 48
    ko = wc.iloc[n_group_games:]
    if len(ko) == (32 if n_group_games == 72 else 16):  # drop the third-place match
        ko = ko.drop(ko.index[-2])
    stages = STAGES if n_group_games == 72 else STAGES[1:]
    best = {t: "Group" for g in groups for t in g}
    games = {}
    for h, a in zip(ko["home_team"], ko["away_team"]):
        for t in (h, a):
            games[t] = games.get(t, 0) + 1
    for t, k in games.items():
        best[t] = stages[min(k - 1, len(stages) - 1)]
    if len(ko):
        final = ko.iloc[-1]
        champ = final["home_team"] if final["home_score"] > final["away_score"] else final["away_team"]
        if final["home_score"] == final["away_score"] and shootouts is not None:
            so = shootouts[(shootouts["date"] == final["date"].strftime("%Y-%m-%d"))
                           & (shootouts["home_team"] == final["home_team"])]
            champ = so["winner"].iloc[0] if len(so) else None
        if champ:
            best[champ] = "Champion"
    return best


def hypothetical_draw(model, hosts, rng, n_teams=48):
    """'If the World Cup started today': hosts plus the best-rated active teams,
    drawn into 12 groups from 4 pots by Elo (no confederation rules)."""
    active = sorted(model.attack, key=lambda t: -model.elo.get(t, 0))
    field = list(hosts) + [t for t in active if t not in hosts][: n_teams - len(hosts)]
    pots = [field[k:k + 12] for k in range(0, 48, 12)]  # hosts sit in pot 1
    groups = [[] for _ in range(12)]
    for p in pots:
        for gi, t in enumerate(rng.permutation(p)):
            groups[gi].append(t)
    return groups, field


def format_table(df, top=20):
    cols = [c for c in STAGES if c in df.columns]
    head = f"{'#':>3}  {'team':22s} {'grp':>3} {'Elo':>5}  " + " ".join(f"{c:>8s}" for c in cols)
    lines = [head]
    for k, r in df.head(top).iterrows():
        cells = " ".join(f"{100 * r[c]:7.1f}%" for c in cols)
        lines.append(f"{k + 1:3d}  {r['team']:22s} {r['group']:>3} {r['elo']:5.0f}  {cells}")
    return "\n".join(lines)
