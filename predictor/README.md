# International football match predictor

Predicts the score, the winner and the probabilities for any match between national
teams. It learns from every international match since 1872: World Cups, qualifiers,
Euros, Copa América, AFCON, Asian Cup, Gold Cup, Nations Leagues and friendlies.
See [PROMPT.md](PROMPT.md) for the full spec it was built from.

## Quick start

```bash
pip install -r predictor/requirements.txt
python -m predictor predict Brazil Argentina --knockout   # neutral venue, e.g. World Cup
python -m predictor predict Nigeria Ghana --home          # first team at home
python -m predictor ratings -n 20                         # strongest teams right now
python -m predictor update                                # download latest results + refit
python -m predictor predict --club Liverpool Arsenal      # club game, with injury news
python -m predictor matches                               # every match yesterday + today (clubs too)
python -m predictor evaluate                              # replay every game since 2000, write a report
python -m predictor matches 2026-11-08                    # or any dates
python -m predictor worldcup --today                      # 48-team World Cup with today's ratings
python -m predictor worldcup --replay 2022                # re-run a past World Cup blind
python -m predictor worldcup --groups my_groups.json --hosts Spain Portugal Morocco
python -m predictor backtest                              # check accuracy on 2022+ matches
```

Output includes win/draw/loss %, expected goals, the 5 most likely scores,
over/under 2.5 goals, both teams to score and (with `--knockout`) the chance
to advance after extra time and penalties.

## World Cup simulator

Plays the whole tournament 10,000 times: every group game is given a scoreline (so
goal difference and tie-breaks work), then the knockouts with extra time and penalties.
It handles the 32-team format and the 48-team format (12 groups, top 2 + 8 best thirds).
Hosts get home advantage. Output: each team's chance to reach each round and to win.

In the 48-team format the official round-of-32 pairing table is simplified: qualifiers
are seeded (group winners, then runners-up, then thirds) to keep top teams apart.

`--replay YEAR` refits the model on matches before that tournament only. How the
model did before each World Cup (the champion's pre-tournament rank and odds):

| World Cup | Champion | Model's pre-tournament rank | Model's odds |
|---|---|---|---|
| 2026 | Spain | #1 of 48 | 19.3% |
| 2022 | Argentina | #2 of 32 | 18.9% |
| 2018 | France | #5 of 32 | 5.8% |
| 2014 | Germany | #4 of 32 | 6.4% |

## Club leagues

`matches` also covers the Premier League, La Liga, Bundesliga, Serie A, Ligue 1,
Primeira Liga and Eredivisie, from the open
[openfootball](https://github.com/openfootball/football.json) data (2015 onward).
Each league is rated on its own (no cross-league games in the data), and promoted
clubs start at the rating of the clubs that went down. On 5,081 unseen league games
since July 2024 it picked the right result 52.4% of the time (no model: 43.0%).
Caveat: the open data's scores can lag several weeks behind real life.

## How it works

1. **Elo ratings** – every match updates both teams' ratings. World Cup games move
   ratings the most, friendlies the least; bigger wins move them more.
2. **Dixon–Coles goals model** – each team gets an attack and defence strength
   (recent and competitive games count more), plus home advantage and the Elo gap.
3. **Player factors** (new) – two extra inputs shift each side's expected goals:
   - *squad value*: combined Transfermarkt market value of the team's best 18 (clubs)
     or 23 (national teams) players at the time
   - *availability*: how much of that value is injured on match day; for the Premier
     League this uses live Fantasy Premier League news (injured, suspended, "75% chance")
4. **Draw adjustment** (clubs) – the plain goals model gave too few draws; one fitted
   parameter fixes that.
5. **Score grid** – expected goals become a probability for every scoreline, and
   every other number is read off that grid.

Predictions list the injured and suspended players when known.

## Every game, replayed

`python -m predictor evaluate` predicts every game since 2000 (national teams) and
2017 (seven leagues), each one using only data from before it, and writes
[reports/report.md](reports/report.md) plus a CSV of every prediction next to the
real result. Headline numbers:

| Games | Count | Right result | Exact score | Log loss before → after |
|---|---|---|---|---|
| All internationals 2000–2026 | 25,879 | 59.4% | 13.8% | 0.8776 → 0.8769 |
| World Cup & continental finals | 2,153 | 56.3% | 15.1% | 0.9413 → 0.9401 |
| League games 2017–2026 | 21,316 | 52.9% | 12.7% | 0.9782 → 0.9778 |
| League games 2024–26 (player data exists) | 5,111 | 52.8% | 12.7% | 0.9807 → 0.9786 |

What changed the results, tested on games the model had not seen:

| Change | Effect |
|---|---|
| Squad market value | small gain for national teams (from 2012, when the value data is dense enough); little extra for clubs |
| Injuries / availability | small, consistent gain for both |
| One name per club (the league data spelled some clubs two ways, splitting their ratings) | small gain |
| Draw adjustment (clubs) | small gain; draws were under-predicted (24.0% vs 25.6%) |
| Rest days | dropped: the data has no cup or European games, so rest is misread |
| Separate home advantage per league, faster/slower Elo | no gain, not used |

**Against bookmakers** (Premier League 2019–2026, 2,594 games): model 53.4% right,
log loss 0.983; bookmakers' closing odds 55.0%, 0.964. Bookmakers still win: they
see confirmed line-ups, team news and money from people with inside knowledge.

The percentages are well calibrated: of games where it gives a home win 50–65%,
57% end in a home win (see the report).

## Data sources

| Data | Source | Up to |
|---|---|---|
| International results | [martj42/international_results](https://github.com/martj42/international_results) (CC0) | refreshed by `update` |
| League results & fixtures | [openfootball/football.json](https://github.com/openfootball/football.json) | refreshed by `update` (can lag weeks) |
| Live Premier League results, injury & suspension news | [oladeeayo/pl-catalogue](https://github.com/oladeeayo/pl-catalogue) (FPL + Transfermarkt) | refreshed by `update` |
| Player market values, injuries, squads | [salimt/football-datasets](https://github.com/salimt/football-datasets) (Transfermarkt), shrunk into `data/players/` by `players/build_tables.py` | values Sep 2025, injuries Sep 2026, squads Sep 2025 |
| Premier League odds (benchmark only) | [AnishKhetani/premier-league-data](https://github.com/AnishKhetani/premier-league-data) | May 2026 |

## Limits

- Outside the Premier League, squads are from September 2025 and injury news stops at
  the end of September 2026, so for games after mid-October 2026 the model ignores
  injuries there (it says so) and squad values miss summer 2026 transfers.
- Confirmed line-ups, xG history, weather, referees and motivation are not used.
  None of these are available from sources this tool can reach.
- Football is noisy: about 1 in 2 league games and 2 in 5 internationals do not
  go the way the favourite expects, even for bookmakers.
