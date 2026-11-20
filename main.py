"""GTA VI trophy rarity forecast (make.fun: gta6-trophy-rarity-forecast-01).

Reads one case file per run from the input directory (every ``*.in`` file, in
name order) and writes ``<id>.out`` JSON forecasts: for each trophy, five
quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95) of its Exophase earn rate, in
percent, at the case's target time.  Python 3.13 standard library only, no
network, fully deterministic.

Model
-----
Work on the logit scale, the scale the pinball loss is scored on.  Take each
trophy's earn-rate series from the case history and collapse runs of
identical reads (Exophase refreshes its page in steps, so consecutive reads
often repeat one page state; keeping only the latest read of each run stops
a long stale stretch from dominating the fit).  Against the log-time axis
``x = ln(1 + hours since release)`` -- growth is fastest right after launch
and settles as the week goes on -- fit two recency-weighted least-squares
lines per trophy (a read's weight halves every 15 hours of age, so the
recent slope, not the launch-night slope, drives the forecast):

* a direct line, ``logit(rate) = a + b*x``;
* a decomposition line, ``ln(earners + 0.5) = a + b*x`` with
  ``earners = rate/100 * players_tracked``, read against a matching fit of
  the tracked pool's own growth, which separates "players keep earning
  this" from "the pool keeps diluting the rate" (a refresh that adds many
  new tracked players pushes most rates down even as earners rise).

Each slope is shrunk toward the pooled slope of all trophies (partial
pooling) and the extrapolation past the last observed state is damped --
only part of the fitted movement is taken, the rest is persistence, since
a log-time line read far past its data overstates trends that are still
steepening.  The two reads are averaged in logit space, and trophies with
too little usable history fall back to persistence (their last rate).

The spread combines the fits' residual variance inflated by the prediction
leverage of extrapolating to the target, a random-walk term growing with
the square root of the horizon, and a refresh term: a share of the typical
logit jump between page states in this case's history -- or, when the pool
never changed in view so no refresh has been seen yet, a prior scale for
the first refresh that is still to come.  A trophy every tracked player
holds in every read gets a narrowed spread.  Quantiles are
``inv_logit(mu + z_tau * sigma)``.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [NormalDist().inv_cdf(tau) for tau in LEVELS]

# The scorer clamps percents into this range before taking logits; mirror it.
PCT_LO = 0.01
PCT_HI = 99.99
# Random-walk volatility of a rate's logit, per square-root day.
RW_SIGMA = 0.45
# Spread used when a trophy has no fittable trend (logit units, before the
# random-walk term is added).
PRIOR_VAR = 0.25
SIGMA_MIN = 0.05
SIGMA_MAX = 3.0
# Exophase refreshes its counts in steps: between two page states a rate can
# jump.  The spread carries a share of the typical jump size seen in the
# case's own history.
JUMP_FACTOR = 0.75
# If the tracked pool never changed in view, no refresh has been seen yet and
# one is certainly coming: price that ignorance with this logit scale.
PENDING_REFRESH_SIGMA = 0.85
# A trophy every tracked player holds in every read can only drift a little.
PINNED_FACTOR = 0.35
# Extrapolation damping: a line fitted on log-time, read far past the data,
# overstates any trend that is still steepening (the tracked pool grows
# near-exponentially in real time early on).  Only this share of the fitted
# movement past the last observed state is taken; the rest is persistence.
DAMP = 0.30
# Fits are recency-weighted: what a rate did in the last day or so says more
# about its next week than its launch-night slope does.
RECENCY_HALF_LIFE_HOURS = 15.0
# The median never moves further than this from the last observed rate
# (logit units): a leash against a wild fit on a strange history.
MAX_MEDIAN_MOVE = 1.2


def clamp_pct(x: float) -> float:
    return min(PCT_HI, max(PCT_LO, x))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(y: float) -> float:
    return 100.0 / (1.0 + math.exp(-y))


def round4(x: float) -> float:
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_utc(text: str) -> float:
    """Epoch seconds for an ISO-8601 UTC timestamp."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def release_epoch(case: dict) -> float:
    """Epoch seconds of 00:00 UTC on the game's release date."""
    day = case["game"]["release_date"]
    dt = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def log_hours(epoch: float, release: float) -> float:
    """ln(1 + hours since release), guarded against pre-release reads."""
    hours = (epoch - release) / 3600.0
    return math.log(max(0.05, 1.0 + hours))


def ols(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least-squares y = a + b*x.  Returns (a, b, resid_var,
    x_mean, sxx) with means and sums weighted by ws, and resid_var the
    weighted mean squared residual."""
    total = sum(ws)
    x_mean = sum(w * x for w, x in zip(ws, xs)) / total
    y_mean = sum(w * y for w, y in zip(ws, ys)) / total
    sxx = sum(w * (x - x_mean) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0.0:
        return y_mean, 0.0, 0.0, x_mean, sxx
    sxy = sum(w * (x - x_mean) * (y - y_mean) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = y_mean - b * x_mean
    resid = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    resid_var = resid / total
    return a, b, resid_var, x_mean, sxx


def recency_weights(xs: list[float]) -> list[float]:
    """Weight per point: halved every RECENCY_HALF_LIFE_HOURS of wall time
    before the newest point.  x is ln(1 + hours since release), so a point's
    age in hours is exp(x_newest) - exp(x)."""
    newest = max(xs)
    return [
        0.5 ** ((math.exp(newest) - math.exp(x)) / RECENCY_HALF_LIFE_HOURS)
        for x in xs
    ]


def trophy_states(case: dict, index: int):
    """Distinct page states for one trophy: (x, rate, players) with runs of
    identical (players, rate) collapsed to their latest read, plus the last
    observed rate."""
    release = release_epoch(case)
    states: list[tuple[float, float, float]] = []
    last_rate = None
    prev_key = None
    for read in case["history"]:
        rate = read["rates"][index]
        if rate is None:
            continue
        last_rate = rate
        key = (read["players_tracked"], rate)
        x = log_hours(parse_utc(read["at"]), release)
        if key == prev_key and states:
            states[-1] = (x, rate, read["players_tracked"])
        else:
            states.append((x, rate, read["players_tracked"]))
            prev_key = key
    return states, last_rate


def fit_line(xs: list[float], ys: list[float]):
    """Recency-weighted OLS fit packaged for shrinkage and extrapolation,
    or None."""
    if len(xs) < 3:
        return None
    ws = recency_weights(xs)
    a, b, resid_var, x_mean, sxx = ols(xs, ys, ws)
    if sxx <= 0.0:
        return None
    total = sum(ws)
    y_mean = sum(w * y for w, y in zip(ws, ys)) / total
    return {
        "b": b,
        "resid_var": resid_var,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "x_last": max(xs),
        "sxx": sxx,
        "n": len(xs),
        "se2": resid_var / sxx,
    }


def shrunk_value(fit: dict, target_x: float, pool_mean: float, pool_var: float) -> float:
    """Read a fit at the target after shrinking its slope toward the pooled
    slope of all trophies (partial pooling: noisy per-trophy slopes borrow
    strength from the field, steady ones keep their own) and damping the
    extrapolation: past the last observed state only DAMP of the fitted
    movement is taken."""
    if pool_var > 0.0:
        w = pool_var / (pool_var + fit["se2"])
    else:
        w = 0.0
    b = w * fit["b"] + (1.0 - w) * pool_mean
    a = fit["y_mean"] - b * fit["x_mean"]
    x_eff = fit["x_last"] + DAMP * (target_x - fit["x_last"])
    return a + b * x_eff


def pool_stats(fits: list) -> tuple[float, float]:
    """Mean and variance of the slopes of the fits that exist."""
    slopes = [f["b"] for f in fits if f is not None]
    if not slopes:
        return 0.0, 0.0
    mean = sum(slopes) / len(slopes)
    var = sum((s - mean) ** 2 for s in slopes) / len(slopes)
    return mean, var


def pred_var_of(fit: dict, target_x: float) -> float:
    leverage = 1.0 + 1.0 / fit["n"] + (target_x - fit["x_mean"]) ** 2 / fit["sxx"]
    return fit["resid_var"] * leverage


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    release = release_epoch(case)
    target_x = log_hours(parse_utc(case["target"]), release)
    days = case["horizon_minutes"] / 1440.0
    rw_var = RW_SIGMA * RW_SIGMA * days

    # How many tracked players will there be at the target?  Fit the growth
    # of the tracked pool itself (log-time: fast early growth that settles).
    pool_states: list[tuple[float, float]] = []
    prev_players = None
    for read in case["history"]:
        players = read["players_tracked"]
        x = log_hours(parse_utc(read["at"]), release)
        if players == prev_players and pool_states:
            pool_states[-1] = (x, math.log(max(1.0, players)))
        else:
            pool_states.append((x, math.log(max(1.0, players))))
            prev_players = players
    pool_fit = fit_line([p[0] for p in pool_states], [p[1] for p in pool_states])
    last_players = case["history"][-1]["players_tracked"]
    if pool_fit is not None:
        ln_pool = shrunk_value(pool_fit, target_x, pool_fit["b"], 0.0)
        pool_hat = math.exp(ln_pool)
        pool_hat = min(50.0 * last_players, max(0.5 * last_players, pool_hat))
    else:
        pool_hat = float(last_players)

    # First pass: per trophy, a direct fit of logit(rate) on log-time and a
    # decomposition fit of log(earners + 0.5) on log-time, where earners are
    # the rate times the tracked pool.  The decomposition separates "players
    # keep earning this" from "the pool keeps diluting the rate".
    per_trophy = []
    for i in range(len(case["trophies"])):
        states, last_rate = trophy_states(case, i)
        xs = [s[0] for s in states]
        direct = fit_line(xs, [logit_pct(s[1]) for s in states])
        earners = fit_line(xs, [math.log(s[1] / 100.0 * s[2] + 0.5) for s in states])
        per_trophy.append(
            {"states": states, "last": last_rate, "direct": direct, "earners": earners}
        )

    direct_pool = pool_stats([p["direct"] for p in per_trophy])
    earners_pool = pool_stats([p["earners"] for p in per_trophy])

    # How big are the steps between page states?  Pool every trophy's jumps
    # (in logit units) and take the median as the case's jump scale.  Also
    # note whether the tracked pool ever changed: if it never did, the page
    # has not refreshed in view and a first refresh is still to come.
    jumps: list[float] = []
    refreshed = False
    prev_players = None
    for read in case["history"]:
        if prev_players is not None and read["players_tracked"] != prev_players:
            refreshed = True
        prev_players = read["players_tracked"]
    for info in per_trophy:
        states = info["states"]
        for before, after in zip(states, states[1:]):
            if before[1] != after[1]:
                jumps.append(abs(logit_pct(after[1]) - logit_pct(before[1])))
    if jumps:
        jumps.sort()
        jump_scale = jumps[len(jumps) // 2]
        jump_var = (JUMP_FACTOR * jump_scale) ** 2 * min(1.0, days)
    elif not refreshed:
        jump_var = PENDING_REFRESH_SIGMA**2 * min(1.0, 2.0 * days)
    else:
        jump_var = 0.0

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        info = per_trophy[i]
        last_rate = info["last"]
        if last_rate is None:
            # Never observed: say nothing is earned, with a modest spread.
            mu, pred_var = logit_pct(0.0), PRIOR_VAR
        else:
            mus: list[float] = []
            pvars: list[float] = []
            if info["direct"] is not None:
                mus.append(shrunk_value(info["direct"], target_x, *direct_pool))
                pvars.append(pred_var_of(info["direct"], target_x))
            if info["earners"] is not None:
                ln_e = shrunk_value(info["earners"], target_x, *earners_pool)
                earners_hat = max(0.0, math.exp(ln_e) - 0.5)
                rate_dec = clamp_pct(100.0 * earners_hat / pool_hat)
                mus.append(logit_pct(rate_dec))
                pvars.append(pred_var_of(info["earners"], target_x))
            if mus:
                mu = sum(mus) / len(mus)
                pred_var = sum(pvars) / len(pvars)
            else:
                mu, pred_var = logit_pct(last_rate), PRIOR_VAR
            anchor = logit_pct(last_rate)
            mu = min(anchor + MAX_MEDIAN_MOVE, max(anchor - MAX_MEDIAN_MOVE, mu))
        sigma = min(SIGMA_MAX, max(SIGMA_MIN, math.sqrt(pred_var + rw_var + jump_var)))
        states = info["states"]
        if states and all(s[1] >= 100.0 for s in states):
            sigma *= PINNED_FACTOR
        qs = []
        prev = 0.0
        for z in Z:
            q = inv_logit_pct(mu + z * sigma)
            q = min(100.0, max(0.0, round4(q)))
            q = max(q, prev)  # keep the row non-decreasing after rounding
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
