"""Command line: python -m predictor <command> ..."""
import argparse

from .data import update_data
from .model import get_model
from .predict import format_prediction, predict_match


def main():
    ap = argparse.ArgumentParser(prog="python -m predictor",
                                 description="International football match predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="predict one match")
    p.add_argument("team_a")
    p.add_argument("team_b")
    p.add_argument("--home", action="store_true",
                   help="team_a plays at home (default: neutral venue, as at a World Cup)")
    p.add_argument("--knockout", action="store_true",
                   help="also show chance to advance after extra time and penalties")

    r = sub.add_parser("ratings", help="show the strongest teams")
    r.add_argument("-n", type=int, default=25)

    b = sub.add_parser("backtest", help="measure accuracy on past matches")
    b.add_argument("--start", default="2022-01-01")

    sub.add_parser("update", help="download the latest results and refit")

    args = ap.parse_args()
    if args.cmd == "predict":
        m = get_model()
        print(format_prediction(predict_match(m, args.team_a, args.team_b,
                                              neutral=not args.home, knockout=args.knockout)))
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
        print("Model refitted.")


if __name__ == "__main__":
    main()
