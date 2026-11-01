# The improved prompt

The original request was: *"make an AI model that uses football teams to predict the
score of the match and who will win, with the probability, based on all competitions."*

That request was rewritten as the spec below, and the code in this folder was built from it.

---

> **Build a football match predictor for national teams, trained on every recorded
> international match (World Cup, World Cup qualifiers, Euros, Copa América, AFCON,
> Asian Cup, Gold Cup, Nations Leagues, friendlies, and so on).**
>
> **Data**
> - Use the open `martj42/international_results` dataset (every international match
>   since 1872, with tournament, venue and neutral-ground flags, plus penalty shootouts).
> - Provide a command to refresh the data so predictions stay current.
>
> **Model** (use proven statistical methods rather than a black box, and make every
> probability come from one consistent score distribution)
> 1. **Elo ratings** (World Football Elo style): K-factor depends on how important the
>    competition is (World Cup > continental finals > qualifiers > friendlies), it grows
>    with goal margin, and the home side gets a bonus.
> 2. **Dixon–Coles Poisson goals model**: each team gets an attack and a defence
>    strength, plus a home-advantage term and the pre-match Elo gap as an extra input.
>    - Recent matches count more (time decay); friendlies count less.
>    - Ridge regularisation pulls teams with little data toward their Elo.
>    - Dixon–Coles correction for low scores (0-0, 1-0, 0-1, 1-1).
> 3. From the predicted goal rates, build the full scoreline probability grid and
>    derive everything from it:
>    - win / draw / loss probabilities
>    - expected goals for each side and the most likely scorelines
>    - over/under 2.5 goals and both-teams-to-score
>    - for knockout games: chance to advance, including extra time and penalties
>
> **Validation**
> - Backtest honestly: train only on matches before a cutoff date, then predict
>   every later match. Report accuracy, log loss, Brier score and ranked probability
>   score, compared with a "base rates only" baseline and with Elo alone.
>
> **Interface**
> - A command-line tool: `predict TEAM_A TEAM_B [--neutral] [--knockout]`,
>   `ratings`, `backtest`, `update`.
> - Accept team names in any letter case and suggest close matches for typos.
>
> **Be honest about limits**: football is noisy. Even good models pick the right
> result only about 55–60% of the time. The model knows nothing about injuries,
> line-ups or form inside a match. Present the output as probabilities, not certainties.
