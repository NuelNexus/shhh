# Every game, replayed
Each game is predicted using only results and player data from before it (the model is refitted every few months as the replay moves forward). **Before** = Elo + Dixon-Coles goals model. **After** = adds squad market value, injuries, merged club names and (clubs) a draw adjustment. Log loss: lower is better; it rewards confident correct predictions and punishes confident wrong ones.

## National teams (2000-01-04 to 2026-10-06)

| games | n | right result | exact score | log loss before | log loss after | improvement |
|---|---|---|---|---|---|---|
| All internationals | 25879 | 59.4% | 13.8% | 0.8776 | 0.8769 | +0.08% |
| World Cup & continental finals | 2153 | 56.3% | 15.1% | 0.9413 | 0.9401 | +0.14% |
| FIFA World Cup only | 488 | 58.8% | 13.9% | 0.9408 | 0.9398 | +0.11% |
| Qualifiers | 10034 | 64.9% | 14.3% | 0.7917 | 0.7907 | +0.13% |
| Friendlies | 8406 | 54.4% | 13.7% | 0.9473 | 0.9469 | +0.04% |

By year:

| games | n | right result | exact score | log loss before | log loss after | improvement |
|---|---|---|---|---|---|---|
| 2000 | 1048 | 60.1% | 13.3% | 0.8631 | 0.8631 | -0.00% |
| 2001 | 1032 | 57.8% | 14.6% | 0.8707 | 0.8707 | -0.00% |
| 2002 | 768 | 55.5% | 15.4% | 0.9441 | 0.9441 | -0.00% |
| 2003 | 948 | 61.9% | 11.5% | 0.8593 | 0.8593 | +0.00% |
| 2004 | 1081 | 57.5% | 12.1% | 0.8914 | 0.8914 | -0.00% |
| 2005 | 807 | 60.2% | 13.1% | 0.8673 | 0.8673 | -0.00% |
| 2006 | 842 | 59.3% | 17.1% | 0.8621 | 0.8621 | +0.00% |
| 2007 | 988 | 56.9% | 13.0% | 0.8884 | 0.8884 | -0.00% |
| 2008 | 1101 | 58.9% | 14.2% | 0.8865 | 0.8865 | -0.00% |
| 2009 | 925 | 60.3% | 12.8% | 0.8796 | 0.8796 | +0.00% |
| 2010 | 863 | 55.5% | 14.6% | 0.9286 | 0.9286 | -0.00% |
| 2011 | 1119 | 59.9% | 15.1% | 0.8571 | 0.8571 | +0.00% |
| 2012 | 1025 | 57.4% | 12.5% | 0.8990 | 0.8975 | +0.17% |
| 2013 | 963 | 59.4% | 14.7% | 0.8860 | 0.8857 | +0.04% |
| 2014 | 856 | 57.0% | 12.4% | 0.9404 | 0.9393 | +0.12% |
| 2015 | 1039 | 61.3% | 14.1% | 0.8739 | 0.8747 | -0.09% |
| 2016 | 920 | 61.7% | 13.8% | 0.8643 | 0.8638 | +0.06% |
| 2017 | 924 | 55.7% | 13.7% | 0.9119 | 0.9112 | +0.08% |
| 2018 | 935 | 57.3% | 15.2% | 0.9136 | 0.9117 | +0.21% |
| 2019 | 1156 | 63.0% | 12.5% | 0.8316 | 0.8298 | +0.22% |
| 2020 | 356 | 55.3% | 9.6% | 0.9558 | 0.9512 | +0.49% |
| 2021 | 1115 | 64.8% | 14.7% | 0.7874 | 0.7853 | +0.27% |
| 2022 | 970 | 59.0% | 13.5% | 0.9099 | 0.9092 | +0.07% |
| 2023 | 1064 | 61.1% | 14.9% | 0.8579 | 0.8565 | +0.16% |
| 2024 | 1231 | 58.8% | 14.5% | 0.8843 | 0.8831 | +0.14% |
| 2025 | 1002 | 62.4% | 14.0% | 0.8123 | 0.8099 | +0.29% |
| 2026 | 801 | 61.8% | 12.2% | 0.8679 | 0.8657 | +0.24% |

## Club leagues (2017-08-04 to 2026-10-25)

| games | n | right result | exact score | log loss before | log loss after | improvement |
|---|---|---|---|---|---|---|
| All league games | 21316 | 52.9% | 12.7% | 0.9782 | 0.9778 | +0.05% |
| Bundesliga | 2790 | 51.3% | 11.9% | 0.9969 | 0.9964 | +0.05% |
| Eredivisie | 2436 | 53.9% | 11.7% | 0.9539 | 0.9529 | +0.11% |
| La Liga | 3479 | 51.9% | 14.4% | 0.9920 | 0.9923 | -0.03% |
| Ligue 1 | 3141 | 50.9% | 11.6% | 1.0055 | 1.0057 | -0.02% |
| Premier League | 3500 | 53.9% | 11.5% | 0.9716 | 0.9714 | +0.02% |
| Primeira Liga | 2510 | 54.4% | 13.7% | 0.9440 | 0.9439 | +0.01% |
| Serie A | 3460 | 54.0% | 13.6% | 0.9732 | 0.9712 | +0.20% |

By season (player data only exists from 2024-25, so earlier seasons barely change):

| games | n | right result | exact score | log loss before | log loss after | improvement |
|---|---|---|---|---|---|---|
| 2017-18 | 1826 | 53.9% | 13.1% | 0.9691 | 0.9691 | -0.00% |
| 2018-19 | 2438 | 52.0% | 12.2% | 0.9758 | 0.9759 | -0.00% |
| 2019-20 | 2061 | 52.9% | 11.7% | 0.9894 | 0.9895 | -0.00% |
| 2020-21 | 2640 | 52.0% | 12.4% | 0.9905 | 0.9905 | -0.00% |
| 2021-22 | 2438 | 52.1% | 13.8% | 0.9805 | 0.9806 | -0.00% |
| 2022-23 | 2438 | 54.1% | 11.9% | 0.9728 | 0.9731 | -0.03% |
| 2023-24 | 2364 | 54.1% | 13.3% | 0.9620 | 0.9614 | +0.06% |
| 2024-25 | 2344 | 53.0% | 13.0% | 0.9753 | 0.9730 | +0.24% |
| 2025-26 | 2362 | 52.3% | 12.6% | 0.9853 | 0.9840 | +0.13% |
| 2026-27 | 405 | 52.6% | 12.6% | 0.9861 | 0.9844 | +0.18% |

## Are the percentages honest? (all games, model after)

When the model says 60%, does it happen about 60% of the time?

| outcome | predicted range | games | predicted | happened |
|---|---|---|---|---|
| home win | (0.0, 0.2] | 6944 | 11.8% | 13.5% |
| home win | (0.2, 0.35] | 9792 | 27.9% | 29.4% |
| home win | (0.35, 0.5] | 11758 | 42.3% | 43.1% |
| home win | (0.5, 0.65] | 8792 | 57.0% | 57.3% |
| home win | (0.65, 0.8] | 5924 | 72.0% | 72.9% |
| home win | (0.8, 1.0] | 3985 | 88.9% | 88.2% |
| draw | (0.0, 0.2] | 11469 | 13.0% | 12.9% |
| draw | (0.2, 0.35] | 35416 | 26.8% | 27.5% |
| draw | (0.35, 0.5] | 310 | 36.2% | 34.2% |
| away win | (0.0, 0.2] | 16023 | 10.6% | 10.2% |
| away win | (0.2, 0.35] | 13849 | 27.3% | 25.6% |
| away win | (0.35, 0.5] | 8746 | 41.8% | 40.1% |
| away win | (0.5, 0.65] | 4946 | 56.7% | 54.3% |
| away win | (0.65, 0.8] | 2341 | 71.3% | 68.6% |
| away win | (0.8, 1.0] | 1290 | 89.1% | 87.4% |

## Against the bookmakers

Premier League games 2019-08-09 to 2026-05-24 with average market odds:

| predictor | games | right result | log loss |
|---|---|---|---|
| Model before | 2594 | 53.0% | 0.9830 |
| Model after | 2594 | 53.4% | 0.9829 |
| Bookmakers, opening odds | 2594 | 54.4% | 0.9681 |
| Bookmakers, closing odds | 2594 | 55.0% | 0.9641 |

## Biggest surprises of the last two years (leagues and competitive internationals)

| date | competition | result | model's chance of that result |
|---|---|---|---|
| 2024-11-30 | Primeira Liga | Sporting Clube de Portugal 0-1 CD Santa Clara | 1.8% |
| 2025-08-30 | Eredivisie | PSV 0-2 Telstar 1963 | 2.4% |
| 2025-03-08 | Bundesliga | FC Bayern München 2-3 VfL Bochum 1848 | 3.0% |
| 2026-01-24 | Bundesliga | FC Bayern München 1-2 FC Augsburg | 3.4% |
| 2024-11-30 | La Liga | FC Barcelona 1-2 UD Las Palmas | 4.1% |
| 2026-02-13 | Eredivisie | FC Volendam 2-1 PSV | 4.7% |
| 2024-11-01 | Ligue 1 | AS Monaco FC 0-1 Angers SCO | 4.9% |
| 2025-03-21 | FIFA World Cup qualification | South Africa 0-3 Lesotho | 5.1% |
| 2026-05-10 | Primeira Liga | AVS 3-1 FC Porto | 5.1% |
| 2025-01-18 | Eredivisie | PEC Zwolle 3-1 PSV | 5.4% |
| 2026-05-02 | Bundesliga | FC Bayern München 3-3 1. FC Heidenheim 1846 | 5.4% |
| 2025-03-21 | FIFA World Cup qualification | Guinea 0-0 Somalia | 5.5% |
