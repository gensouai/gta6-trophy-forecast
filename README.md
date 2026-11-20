# Forecast how many GTA VI players earn each trophy in week one

Entry for the [make.fun bounty "Forecast how many GTA VI players earn each trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gensouai/gta6-trophy-forecast/rank.svg)
![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/gensouai/gta6-trophy-forecast/reward.svg)

`main.py` (Python 3.13, standard library only, no network) reads each case
file in the input directory and writes a JSON forecast of trophy earn-rate
quantiles (q05/q25/q50/q75/q95, percent) per trophy, deterministically —
two runs produce byte-identical output. The model works in logit space with
recency-weighted fits, shrinkage toward the pooled trend, and a
refresh-aware uncertainty spread.

License: MIT (see `LICENSE`).
