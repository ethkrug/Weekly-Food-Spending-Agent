"""Live NYC subway ridership from data.ny.gov (Socrata), for the nyc_ridership tool.

This is the only tool that reaches the internet, and the only one that sends a query language
(SoQL) anywhere -- so it is the most locked down:

  * The model never writes SoQL. It picks a station set, a year, a season and an hour window from
    fixed lists; this module builds the query from constants. Station IDs are integers from a table
    below, dates are computed, nothing the model types is pasted into the query.
  * Only GET requests, only to data.ny.gov, only to the three ridership datasets below.
  * The query is aggregated server-side (one row per day, about 60 rows) and capped with $limit.
  * Timeout, one retry, a response-size cap and a schema check on what comes back: anything odd is
    an error the model can read, never raw text from the internet.
  * Results are cached for 12 hours, and the app caps calls per research run.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta

import httpx
import numpy as np

HOST = "https://data.ny.gov"
DATASETS = {  # year range -> Socrata dataset id
    (2017, 2019): "t69i-h2me",
    (2020, 2024): "wujg-7c2s",
    (2025, 2100): "5wq4-mkjj",
}
STATION_SETS = {
    # Station complex IDs are stable across years even when names change.
    "office_11": [610, 611, 225, 276, 612, 607, 318, 164, 628, 333, 413],
    "systemwide": None,
}
STATION_NAMES = ("Grand Central-42 St, Times Sq-42 St, 47-50 Sts-Rockefeller Ctr, 5 Av/53 St, Lexington Av/51-53 Sts, "
                 "34 St-Herald Sq, 34 St-Penn Station (two complexes), Fulton St, Wall St (two stations)")
SEASONS = {"fall": ((9, 15), (11, 14)), "spring": ((3, 1), (4, 30))}
HOURS = {"all_day": None, "morning_commute": (7, 9), "lunch": (11, 13), "evening_commute": (16, 18)}
TIMEOUT = 45
MAX_BYTES = 200_000
CACHE_TTL = 12 * 3600
_cache: dict[tuple, tuple[float, dict]] = {}

WEEKDAYS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


class RidershipError(RuntimeError):
    pass


def _federal_holidays(year: int) -> set[date]:
    def nth_monday(month, n):
        d = date(year, month, 1)
        d += timedelta(days=(0 - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)

    vets = date(year, 11, 11)
    vets_observed = vets - timedelta(days=1) if vets.weekday() == 5 else vets + timedelta(days=1) if vets.weekday() == 6 else vets
    return {nth_monday(10, 2), vets, vets_observed}  # Columbus/Indigenous Peoples Day, Veterans Day


def _dataset(year: int) -> str:
    for (lo, hi), ds in DATASETS.items():
        if lo <= year <= hi:
            return ds
    raise RidershipError(f"No ridership data for {year}; data starts in 2017.")


def _fetch_daily(year: int, station_set: str, season: str, hours: str) -> dict[date, float]:
    (m0, d0), (m1, d1) = SEASONS[season]
    start, end = date(year, m0, d0), date(year, m1, d1)
    where = [f"transit_timestamp between '{start.isoformat()}T00:00:00' and '{end.isoformat()}T23:59:59'"]
    ids = STATION_SETS[station_set]
    if ids:
        where.append("station_complex_id in (" + ",".join(f"'{int(i)}'" for i in ids) + ")")
    if HOURS[hours]:
        h0, h1 = HOURS[hours]
        where.append(f"date_extract_hh(transit_timestamp) between {int(h0)} and {int(h1)}")
    params = {
        "$select": "date_trunc_ymd(transit_timestamp) as d, sum(ridership) as r",
        "$where": " and ".join(where),
        "$group": "d",
        "$order": "d",
        "$limit": "500",
    }
    headers = {"Accept": "application/json", "User-Agent": "food-week-deep-research/0.1"}
    if os.environ.get("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]
    url = f"{HOST}/resource/{_dataset(year)}.json"
    last_error = None
    for attempt in range(2):
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=TIMEOUT, follow_redirects=False)
            if r.status_code != 200:
                raise RidershipError(f"data.ny.gov returned HTTP {r.status_code}")
            if len(r.content) > MAX_BYTES:
                raise RidershipError("data.ny.gov response was unexpectedly large")
            rows = r.json()
            break
        except (httpx.HTTPError, ValueError, RidershipError) as e:
            last_error = e
            time.sleep(1.5)
    else:
        raise RidershipError(f"Could not reach data.ny.gov: {last_error}")
    if not isinstance(rows, list):
        raise RidershipError("Unexpected response shape from data.ny.gov")
    out = {}
    for row in rows:
        try:
            out[date.fromisoformat(str(row["d"])[:10])] = float(row["r"])
        except (KeyError, TypeError, ValueError):
            raise RidershipError("Unexpected row in data.ny.gov response")
    return out


def weekday_shares(year: int, station_set: str, season: str = "fall", hours: str = "all_day") -> dict:
    """Each weekday's average entries as a percent of the Mon-Fri average, with day-to-day SEs."""
    if station_set not in STATION_SETS:
        raise ValueError(f"station_set must be one of {list(STATION_SETS)}")
    if season not in SEASONS:
        raise ValueError(f"season must be one of {list(SEASONS)}")
    if hours not in HOURS:
        raise ValueError(f"hours must be one of {list(HOURS)}")
    this_year = date.today().year
    if not 2017 <= int(year) <= this_year:
        raise ValueError(f"year must be between 2017 and {this_year}")
    key = (int(year), station_set, season, hours)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    daily = _fetch_daily(int(year), station_set, season, hours)
    holidays = _federal_holidays(int(year))
    by_day: dict[int, list[float]] = {i: [] for i in WEEKDAYS}
    for d, v in daily.items():
        if d.weekday() in WEEKDAYS and d not in holidays:
            by_day[d.weekday()].append(v)
    if min(len(v) for v in by_day.values()) < 3:
        raise RidershipError(f"Too few days of data for {season} {year} (the window may not have happened yet).")
    means = {k: float(np.mean(v)) for k, v in by_day.items()}
    ses = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in by_day.items()}
    avg = float(np.mean(list(means.values())))
    result = {
        "year": int(year), "station_set": station_set, "season": season, "hours": hours,
        "days_used": {WEEKDAYS[k]: len(v) for k, v in by_day.items()},
        "first_day": min(daily).isoformat(), "last_day": max(daily).isoformat(),
        "avg_entries": {WEEKDAYS[k]: means[k] for k in WEEKDAYS},
        "se_entries": {WEEKDAYS[k]: ses[k] for k in WEEKDAYS},
        "index": {WEEKDAYS[k]: 100 * means[k] / avg for k in WEEKDAYS},
        "weekday_avg": avg,
    }
    _cache[key] = (time.time(), result)
    return result
