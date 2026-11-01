"""Dixon-Coles Poisson goals model with Elo as an extra input.

Expected goals for the home side:
    log(lambda_home) = mu + home_adv * (not neutral) + attack[home] - defence[away] + beta * elo_gap
    log(lambda_away) = mu + attack[away] - defence[home] - beta * elo_gap
where elo_gap = (elo_home - elo_away) / 400, measured before the match and
softly capped so huge mismatches do not predict absurd scores.

Teams with little recent data are pulled toward attack = defence = 0 by a ridge
penalty, so their prediction falls back on Elo.
"""
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from .data import DATA_DIR, FIT_WEIGHT, competition_tier, load_results
from .elo import run_elo, START

MAX_GOALS = 15
GAP_CAP = 2.0  # Elo effect levels off past ~800 points (real data: ~5-6 goals at most)


def elo_gap(elo_home, elo_away):
    g = (elo_home - elo_away) / 400.0
    return GAP_CAP * np.tanh(g / GAP_CAP)
MODEL_PATH = DATA_DIR / "model.pkl"


@dataclass
class Settings:
    window_years: float = 20.0     # only fit on matches this recent
    half_life_years: float = 8.0   # a match this old counts half as much
    ridge: float = 1.0             # pull toward Elo-only prediction
    use_team_strengths: bool = True


@dataclass
class Model:
    settings: Settings
    mu: float = 0.0
    home_adv: float = 0.0
    beta: float = 0.0
    rho: float = 0.0
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)
    elo: dict = field(default_factory=dict)
    trained_until: pd.Timestamp = None
    n_matches: int = 0

    # ---------- fitting ----------
    @classmethod
    def fit(cls, df, cutoff=None, settings=None, pre_elo=None, final_elo=None):
        """Fit on matches strictly before `cutoff` (default: all data)."""
        settings = settings or Settings()
        if pre_elo is None:
            _, pre_h, pre_a = run_elo(df)  # pre-match values only use earlier games
        else:
            pre_h, pre_a = pre_elo
        cutoff = pd.Timestamp(cutoff) if cutoff is not None else df["date"].max() + pd.Timedelta(days=1)

        start = cutoff - pd.Timedelta(days=365.25 * settings.window_years)
        mask = ((df["date"] < cutoff) & (df["date"] >= start)).to_numpy()
        d = df[mask]
        gap = elo_gap(pre_h[mask], pre_a[mask])

        teams = sorted(set(d["home_team"]) | set(d["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        hi = d["home_team"].map(idx).to_numpy()
        ai = d["away_team"].map(idx).to_numpy()
        hg = d["home_score"].to_numpy(float)
        ag = d["away_score"].to_numpy(float)
        home = (~d["neutral"]).to_numpy(float)
        age_years = (cutoff - d["date"]).dt.days.to_numpy() / 365.25
        w = np.exp(-np.log(2) * age_years / settings.half_life_years)
        w *= d["tournament"].map(lambda t: FIT_WEIGHT[competition_tier(t)]).to_numpy()
        n = len(teams)
        use_teams = settings.use_team_strengths

        def unpack(p):
            att = p[3:3 + n] if use_teams else np.zeros(n)
            dfn = p[3 + n:] if use_teams else np.zeros(n)
            return p[0], p[1], p[2], att, dfn

        def loss(p):
            mu, h, b, att, dfn = unpack(p)
            lh = mu + h * home + att[hi] - dfn[ai] + b * gap
            la = mu + att[ai] - dfn[hi] - b * gap
            eh, ea = np.exp(lh), np.exp(la)
            nll = np.sum(w * (eh - hg * lh)) + np.sum(w * (ea - ag * la))
            rh, ra = w * (eh - hg), w * (ea - ag)  # d(nll)/d(log lambda)
            g = [rh.sum() + ra.sum(), (rh * home).sum(), (rh * gap).sum() - (ra * gap).sum()]
            if use_teams:
                g_att = np.bincount(hi, rh, n) + np.bincount(ai, ra, n)
                g_def = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n)
                nll += settings.ridge * (att @ att + dfn @ dfn)
                g_att += 2 * settings.ridge * att
                g_def += 2 * settings.ridge * dfn
                return nll, np.concatenate([g, g_att, g_def])
            return nll, np.array(g)

        p0 = np.zeros(3 + (2 * n if use_teams else 0))
        p0[0] = np.log(max((hg.mean() + ag.mean()) / 2, 0.1))
        res = minimize(loss, p0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        mu, h, b, att, dfn = unpack(res.x)

        # Dixon-Coles low-score correction, fitted after the rates.
        lh = np.exp(mu + h * home + att[hi] - dfn[ai] + b * gap)
        la = np.exp(mu + att[ai] - dfn[hi] - b * gap)

        def rho_nll(r):
            t = np.ones_like(lh)
            t = np.where((hg == 0) & (ag == 0), 1 - lh * la * r, t)
            t = np.where((hg == 0) & (ag == 1), 1 + lh * r, t)
            t = np.where((hg == 1) & (ag == 0), 1 + la * r, t)
            t = np.where((hg == 1) & (ag == 1), 1 - r, t)
            return -np.sum(w * np.log(np.clip(t, 1e-9, None)))

        rho = minimize_scalar(rho_nll, bounds=(-0.2, 0.2), method="bounded").x

        if final_elo is None:
            final_elo, _, _ = run_elo(df[df["date"] < cutoff])
        return cls(
            settings=settings, mu=mu, home_adv=h, beta=b, rho=rho,
            attack=dict(zip(teams, att)), defence=dict(zip(teams, dfn)),
            elo=final_elo, trained_until=cutoff, n_matches=int(mask.sum()),
        )

    # ---------- prediction ----------
    def rates(self, home, away, neutral=True, elo_home=None, elo_away=None):
        eh = self.elo.get(home, START) if elo_home is None else elo_home
        ea = self.elo.get(away, START) if elo_away is None else elo_away
        gap = elo_gap(eh, ea)
        ah, dh = self.attack.get(home, 0.0), self.defence.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defence.get(away, 0.0)
        lh = np.exp(self.mu + (0 if neutral else self.home_adv) + ah - da + self.beta * gap)
        la = np.exp(self.mu + aa - dh - self.beta * gap)
        return lh, la

    def score_grid(self, lh, la):
        g = np.outer(poisson.pmf(np.arange(MAX_GOALS + 1), lh),
                     poisson.pmf(np.arange(MAX_GOALS + 1), la))
        r = self.rho
        g[0, 0] *= 1 - lh * la * r
        g[0, 1] *= 1 + lh * r
        g[1, 0] *= 1 + la * r
        g[1, 1] *= 1 - r
        return g / g.sum()

    def outcome_probs(self, home, away, neutral=True, **kw):
        g = self.score_grid(*self.rates(home, away, neutral, **kw))
        return np.tril(g, -1).sum(), np.trace(g), np.triu(g, 1).sum()

    def save(self, path=MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=MODEL_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)


def get_model(refit=False):
    """Load the saved model, refitting if the data is newer or it is missing."""
    data_file = DATA_DIR / "results.csv"
    if not refit and MODEL_PATH.exists() and data_file.exists() \
            and MODEL_PATH.stat().st_mtime >= data_file.stat().st_mtime:
        return Model.load()
    print("Fitting model on all matches (takes a few seconds)...")
    m = Model.fit(load_results())
    m.save()
    return m
