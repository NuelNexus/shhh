"""One-off script: map each league-data club name to its Transfermarkt club id.

Run:  python -m predictor.players.build_club_ids /path/to/football-datasets
Writes predictor/data/clubs/club_ids.csv (committed, so normal use never needs this).
"""
import difflib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from ..clubs import CLUB_DIR, load_club_matches

TIERS = {  # league code -> Transfermarkt first and second division ids
    "en.1": ("GB1", "GB2"), "es.1": ("ES1", "ES2"), "de.1": ("L1", "L2"),
    "it.1": ("IT1", "IT2"), "fr.1": ("FR1", "FR2"), "pt.1": ("PO1", "PO2"),
    "nl.1": ("NL1", "NL2"),
}
# Names the fuzzy matcher gets wrong, checked by hand.
OVERRIDES = {
    "NEC": 467, "Sport Lisboa e Benfica": 294, "CD Alavés": 1108, "AC Pisa 1909": 4172,
    "SPAL 2013 Ferrara": 2722, "Académico de Viseu FC": 7788, "AVS": 110302,
    "Racing Club de Lens": 826, "Sporting Clube de Braga": 1075, "Sporting Braga": 1075,
    "CD Aves": 3336, "AZ": 1090, "PSV": 383, "AFC Ajax": 610, "Lille OSC": 1082,
    "SV 07 Elversberg": 64, "SC Cambuur": 133, "SBV Excelsior": 798, "SBV Vitesse": 499,
    "FC Twente": 317, "FC Twente '65": 317, "Athletic Club": 621, "Real Sociedad de Fútbol": 681,
    "Real Sociedad": 681, "Belenenses SAD": 2457,
}


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = s.replace("munchen", "munich").replace("internazionale", "inter")
    s = re.sub(r"\b(fc|cf|ac|afc|sc|ssc|as|ss|us|rc|cd|ud|sd|rcd|ca|cfc|sv|tsg|vfb|vfl|bv|ogc|sco"
               r"|fk|sl|gd|cs|sbv|club|de|del|calcio|the|\d+)\b", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(s.split())


def main(tm_root):
    tcs = pd.read_csv(Path(tm_root) / "datalake/transfermarkt/team_competitions_seasons/"
                      "team_competitions_seasons.csv", low_memory=False)
    played, fixtures = load_club_matches()
    games = pd.concat([played, fixtures])
    rows = []
    for code, comps in TIERS.items():
        tm = tcs[tcs["competition_id"].isin(comps)].sort_values("season_id")
        tm = tm.drop_duplicates("club_id", keep="last")
        by_norm = {norm(n): (cid, n) for cid, n in zip(tm["club_id"], tm["team_name"])}
        id_name = dict(zip(tcs["club_id"], tcs["team_name"]))
        for name in sorted(set(games.loc[games["league"] == code, "home_team"])):
            if name in OVERRIDES:
                cid = OVERRIDES[name]
                rows.append((code, name, cid, id_name.get(cid, ""), 1.0))
                continue
            best = difflib.get_close_matches(norm(name), list(by_norm), n=1, cutoff=0.0)[0]
            score = difflib.SequenceMatcher(None, norm(name), best).ratio()
            rows.append((code, name, by_norm[best][0], by_norm[best][1], round(score, 2)))
    out = pd.DataFrame(rows, columns=["league", "name", "tm_id", "tm_name", "match_score"])
    out.to_csv(CLUB_DIR / "club_ids.csv", index=False)
    print(out.sort_values("match_score").head(25).to_string())
    print(f"\n{len(out)} names -> {out['tm_id'].nunique()} clubs")


if __name__ == "__main__":
    main(sys.argv[1])
