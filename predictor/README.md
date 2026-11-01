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
python -m predictor matches                               # every match yesterday + today (clubs too)
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

1. **Elo ratings** – every match since 1872 updates both teams' ratings. World Cup
   games move ratings the most, friendlies the least; bigger wins move them more.
2. **Dixon–Coles goals model** – each team gets an attack and defence strength,
   fitted on the last 20 years (recent games and competitive games count more),
   plus home advantage and the Elo gap. This gives expected goals for each side.
3. **Score grid** – expected goals become a probability for every scoreline
   (0-0 up to 15-15), and every other number is read off that grid.

## How good is it?

Trained only on earlier matches, then tested on 5,068 matches from Jan 2022 to Oct 2026:

| | Correct result | Log loss | Brier |
|---|---|---|---|
| This model | 60.3% | 0.866 | 0.508 |
| Elo only | 60.4% | 0.876 | 0.516 |
| No model (base rates) | 47.7% | 1.051 | 0.634 |

On World Cup and continental finals only it picks the right result 56.9% of the time.
Football is noisy: nobody gets close to 100%. The model doesn't know about injuries,
line-ups, coaches or player form, so treat its output as probabilities, not certainties.

Data: [martj42/international_results](https://github.com/martj42/international_results) (CC0).
