"""Command line: python -m predictor <command> ..."""
import argparse

from .data import update_data
from .model import get_model
from .predict import format_prediction, predict_match


def main():
    ap = argparse.ArgumentParser(prog="python -m predictor",
                                 description="Football match predictor (national teams and European leagues)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="predict one match")
    p.add_argument("team_a")
    p.add_argument("team_b")
    p.add_argument("--home", action="store_true",
                   help="team_a plays at home (default: neutral venue, as at a World Cup)")
    p.add_argument("--knockout", action="store_true",
                   help="also show chance to advance after extra time and penalties")
    p.add_argument("--club", action="store_true",
                   help="club game (team_a at home), e.g. predict --club Liverpool Arsenal")
    p.add_argument("--date", default=None, help="match date for player news (default: today)")

    r = sub.add_parser("ratings", help="show the strongest teams")
    r.add_argument("-n", type=int, default=25)

    b = sub.add_parser("backtest", help="measure accuracy on past matches")
    b.add_argument("--start", default="2022-01-01")

    sub.add_parser("update", help="download the latest results and refit")

    ev = sub.add_parser("evaluate", help="replay every game, compare old vs new model, write a report")
    ev.add_argument("--quick", action="store_true", help="recent seasons only")

    w = sub.add_parser("worldcup", help="simulate a World Cup thousands of times")
    g = w.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", type=int, metavar="YEAR",
                   help="re-run a past World Cup (2014/2018/2022/2026) using only data from before it")
    g.add_argument("--today", action="store_true",
                   help="hypothetical 48-team World Cup with today's ratings (2030 hosts, random draws)")
    g.add_argument("--groups", metavar="FILE",
                   help="JSON list of groups, e.g. [[\"Spain\", \"Japan\", ...], ...] (8 or 12 groups)")
    w.add_argument("--hosts", nargs="*", default=None)
    w.add_argument("-n", type=int, default=10000, help="number of simulations")
    w.add_argument("--top", type=int, default=20)
    w.add_argument("--seed", type=int, default=7)

    mt = sub.add_parser("matches", help="predict every scheduled match on given dates (clubs + national teams)")
    mt.add_argument("dates", nargs="*", help="YYYY-MM-DD, 'today' or 'yesterday' (default: both)")

    args = ap.parse_args()
    if args.cmd == "predict":
        run_predict(args)
    elif args.cmd == "ratings":
        m = get_model()
        rows = []
        for t, e in m.elo.items():
            if t not in m.attack:
                continue  # no recent matches
            lh, la = m.rates(t, "__average__", neutral=True, elo_away=1500)
            rows.append((e, t, lh, la))
        rows.sort(reverse=True)
        print(f"Data up to {m.trained_until.date()} ({m.n_matches} matches in fit)\n")
        print(f"{'#':>3}  {'team':24s} {'Elo':>6s}  goals for/against vs a 1500-Elo side")
        for k, (e, t, lh, la) in enumerate(rows[:args.n], 1):
            print(f"{k:3d}  {t:24s} {e:6.0f}  {lh:.2f} / {la:.2f}")
    elif args.cmd == "backtest":
        from .backtest import run_backtest
        run_backtest(start=args.start)
    elif args.cmd == "update":
        update_data()
        get_model(refit=True)
        from .clubs import update_club_data, get_club_model
        from .live_pl import update_live
        update_club_data()
        try:
            update_live()
        except Exception as e:  # optional source
            print(f"Live Premier League data not updated: {e}")
        get_club_model(refit=True)
        print("Models refitted.")
    elif args.cmd == "evaluate":
        from .report import run_report
        run_report(quick=args.quick)
    elif args.cmd == "worldcup":
        run_worldcup(args)
    elif args.cmd == "matches":
        run_matches(args)


def run_predict(args):
    import datetime as dt
    import pandas as pd
    date = pd.Timestamp(args.date or dt.date.today())
    if args.club:
        from .clubs import club_match_features, get_club_model, resolve_club, load_club_matches
        m = get_club_model()
        home, away = resolve_club(args.team_a, m), resolve_club(args.team_b, m)
        played, _ = load_club_matches()
        feats, missing, source, league = club_match_features(home, away, date, played)
        p = predict_match(m, home, away, neutral=False, knockout=args.knockout,
                          features=feats, league=league, resolve=lambda n, _: n)
        p["missing"], p["source"] = missing, source
        print(format_prediction(p))
        return
    from .features import player_tables
    from .predict import resolve_team
    m = get_model()
    a, b = resolve_team(args.team_a, m), resolve_team(args.team_b, m)
    feats, missing = {}, {}
    t = player_tables()
    if t is not None:
        sa, va = t.nation_features(a, [date])
        sb, vb = t.nation_features(b, [date])
        feats = {"squad_diff": float(sa[0] - sb[0]), "avail_diff": float(va[0] - vb[0])}
        if not t._stale([date])[0]:
            missing = {x: [(n, "") for n, _ in t.missing_players(x, date, kind="nation")] for x in (a, b)}
    p = predict_match(m, a, b, neutral=not args.home, knockout=args.knockout, features=feats)
    p["missing"] = missing
    print(format_prediction(p))


def run_worldcup(args):
    import json
    import numpy as np
    import pandas as pd
    from .data import DATA_DIR, load_results
    from .model import Model
    from .predict import resolve_team
    from .worldcup import (HOSTS, Simulator, actual_finish, format_table,
                           groups_from_results, hypothetical_draw)

    rng = np.random.default_rng(args.seed)
    if args.replay:
        df = load_results()
        groups, wc = groups_from_results(df, args.replay)
        if not groups:
            raise SystemExit(f"No World Cup found in {args.replay}.")
        kickoff = wc["date"].min()
        print(f"Fitting model on matches before {kickoff.date()} only...")
        from .features import add_features
        from .model import NATION_SETTINGS
        model = Model.fit(add_features(df, "nation"), cutoff=kickoff, settings=NATION_SETTINGS)
        hosts = args.hosts if args.hosts is not None else HOSTS.get(args.replay, [])
        res = Simulator(model, hosts, rng, date=kickoff).run(groups, args.n)
        shootouts = pd.read_csv(DATA_DIR / "shootouts.csv")
        actual = actual_finish(wc, groups, shootouts)
        res["actual"] = res["team"].map(actual)
        print(f"\n{args.replay} World Cup, {args.n:,} simulations (hosts: {', '.join(hosts) or 'none'})\n")
        table = format_table(res, args.top).splitlines()
        print(table[0] + "   actual")
        for line, (_, r) in zip(table[1:], res.head(args.top).iterrows()):
            print(f"{line}   {r['actual']}")
        champ = res[res["actual"] == "Champion"]
        if len(champ):
            c = champ.iloc[0]
            rank = champ.index[0] + 1
            print(f"\nActual champion: {c['team']} - the model gave them {100 * c['Champion']:.1f}%"
                  f" (ranked #{rank} of {len(res)} before the tournament).")
        return

    from .model import get_model
    model = get_model()
    if args.groups:
        groups = [[resolve_team(t, model) for t in g] for g in json.load(open(args.groups))]
        hosts = [resolve_team(h, model) for h in (args.hosts or [])]
        res = Simulator(model, hosts, rng, date=pd.Timestamp.today().normalize()).run(groups, args.n)
        print(f"\nCustom World Cup, {args.n:,} simulations\n")
        print(format_table(res, args.top))
        return

    # --today: average over many random draws
    hosts = args.hosts if args.hosts is not None else HOSTS[2030]
    hosts = [resolve_team(h, model) for h in hosts]
    n_draws = 100
    per_draw = max(args.n // n_draws, 1)
    sim = Simulator(model, hosts, rng, date=pd.Timestamp.today().normalize())
    total = None
    for _ in range(n_draws):
        groups, field = hypothetical_draw(model, hosts, rng)
        r = sim.run(groups, per_draw).set_index("team").drop(columns="group")
        total = r if total is None else total.add(r, fill_value=0)
    res = (total / n_draws).reset_index()
    res["elo"] = res["team"].map(model.elo)
    res["group"] = "-"
    res = res.sort_values(["Champion", "Final"], ascending=False).reset_index(drop=True)
    print(f"\nIf a 48-team World Cup started today (data to {model.trained_until.date()}):"
          f" hosts {', '.join(hosts)}, top {48 - len(hosts)} teams by Elo,"
          f" {n_draws} random draws x {per_draw} simulations\n")
    print(format_table(res, args.top))


def run_matches(args):
    import datetime as dt
    import numpy as np
    import pandas as pd
    from .clubs import LEAGUES, club_match_features, get_club_model, load_club_matches
    from .data import DATA_DIR

    today = dt.date.today()
    words = {"today": today, "yesterday": today - dt.timedelta(days=1)}
    dates = [pd.Timestamp(words.get(d, d)) for d in (args.dates or ["yesterday", "today"])]

    nat = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    nat = nat[nat["date"].isin(dates)]
    played, fixtures = load_club_matches()
    from .live_pl import add_live_results
    played, fixtures = add_live_results(played, fixtures)
    clubs = pd.concat([played, fixtures])
    clubs = clubs[clubs["date"].isin(dates)]

    def show(title, games, model, club):
        print(f"\n== {title} ==")
        print(f"  {'date':10s} {'home':24s} {'away':24s} {'home':>5s} {'draw':>5s} {'away':>5s}"
              f"   {'xG':>9s}  {'likely':>6s}  {'result':>6s}")
        for _, g in games.iterrows():
            h, a = g["home_team"], g["away_team"]
            feats, missing, league = {}, {}, None
            if club:
                feats, missing, _, league = club_match_features(h, a, g["date"], played)
            lh, la = model.rates(h, a, bool(g["neutral"]), features=feats, league=league)
            grid = model.score_grid(lh, la)
            pw, pdw, pl = np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()
            i, j = np.unravel_index(grid.argmax(), grid.shape)
            res = "" if pd.isna(g["home_score"]) else f"{int(g['home_score'])}-{int(g['away_score'])}"
            print(f"  {g['date'].date()!s:10s} {h[:24]:24s} {a[:24]:24s} {100*pw:4.0f}% {100*pdw:4.0f}% {100*pl:4.0f}%"
                  f"   {lh:.1f} - {la:.1f}  {i}-{j:<4d}  {res:>6s}")
            for t, out in missing.items():
                if out:
                    print(f"{'':14s}{t[:24]} without: {', '.join(n for n, _ in out[:4])}")

    if len(nat):
        show("International", nat, get_model(), False)
    else:
        print("\nNo national-team matches on these dates.")
    if len(clubs):
        model = get_club_model()
        print(f"\n(Club results up to {played['date'].max().date()}; Premier League includes live data.)")
        for code, name in LEAGUES.items():
            games = clubs[clubs["league"] == code]
            if len(games):
                show(name, games, model, True)
    else:
        print("No club matches found on these dates.")


if __name__ == "__main__":
    main()
