"""The queries Researchers can run. Every one is fixed: the model picks options from short lists, and
code does the weighting, the standard errors and the weekday comparisons.

No tool here takes SQL or free text from the model. DuckDB queries are assembled from column names
in the allow-lists below, and every value is passed as a bound parameter.

Each function returns a Result: the text the Researcher reads, plus a record the app files in the
evidence registry so the Writer can cite it and code can check the citation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "data" / "food_week.duckdb"
WEEKDAYS = {2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri"}
DAY_NAMES = {2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday"}

# --- How a weekday difference is described. Tuned by calibrate_significance.py (see README). ---
# |difference| / SE(difference) at or above CLEAR_Z: "clear"; at or above SUGGESTIVE_Z: "suggestive";
# below: "no clear difference". Writers may say "higher"/"lower" for clear, "leans higher"/"somewhat
# higher" for suggestive, and must not call a no-clear-difference gap a difference.
CLEAR_Z = 2.0
SUGGESTIVE_Z = 1.5

# Standard-error inflation per source, from the shuffled-weekday noise test in calibrate_significance.py:
# with no real weekday effect, CE replicate-weight SEs flagged |z| >= 2 12% of the time (not ~5%), and
# ATUS SEs (which ignore the survey design) 7%. Scaling by these factors brings both back to nominal.
SE_SCALE = {"ce_diary": 1.3, "atus": 1.1, "dcpc": 1.0, "nhanes": 1.0}

# Cells smaller than this are not reported as numbers.
MIN_N = 100
MIN_N_LOCAL = 30  # New York subsamples: reported, but flagged as thin below MIN_N

SE_NOTE = {"ce_diary": "SEs from 44 replicate weights, widened 1.3x (calibrated)",
           "atus": "SEs widened 1.1x for the survey design (calibrated)"}

SOURCE_LINKS = {
    "ce_diary": ("BLS Consumer Expenditure Survey, Diary public-use microdata", "https://www.bls.gov/cex/pumd.htm"),
    "dcpc": ("Federal Reserve Bank of Atlanta, Diary of Consumer Payment Choice",
             "https://www.atlantafed.org/research-and-data/surveys/survey-and-diary-of-consumer-payment-choice"),
    "nhanes": ("CDC NHANES, What We Eat in America dietary recalls", "https://wwwn.cdc.gov/nchs/nhanes/"),
    "atus": ("BLS American Time Use Survey microdata", "https://www.bls.gov/tus/data.htm"),
    "mta": ("MTA Subway Hourly Ridership, data.ny.gov", "https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-2025/5wq4-mkjj"),
    "reference": ("Cited office-attendance reports (Placer.ai, Kastle, Partnership for NYC)", ""),
}


class ToolInputError(ValueError):
    """A bad argument. The message goes back to the model so it can fix the call."""


@dataclass
class Result:
    text: str  # what the Researcher reads
    source: str  # key in SOURCE_LINKS
    params: dict
    caveats: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Loading: one read-only database, opened lazily; a cursor per query
# --------------------------------------------------------------------------------------------

_lock = threading.Lock()
_con: duckdb.DuckDBPyConnection | None = None
_frames: dict[str, pd.DataFrame] = {}


# Tools run in parallel threads (the Agents SDK runs sync tools with asyncio.to_thread), and a DuckDB
# connection refuses a second call while one is running. So: one read-only database handle, and a fresh
# cursor (its own connection to the same database) for every query. DuckDB is also capped to the Cloud
# Run container, since by default it sizes threads and memory to the host machine, not the container.
DUCKDB_CONFIG = {"threads": 2, "memory_limit": "384MB"}


def con() -> duckdb.DuckDBPyConnection:
    """A new cursor on the shared read-only database. Use one per query; never share it across threads."""
    global _con
    with _lock:
        if _con is None:
            if not DB_PATH.exists():
                raise RuntimeError(f"{DB_PATH} is missing: run `uv run --group build build_data.py` first.")
            _con = duckdb.connect(str(DB_PATH), read_only=True, config=DUCKDB_CONFIG)
        return _con.cursor()


# At most two queries at once: each takes well under a second, and running eight together (five parallel
# Researchers) pushed memory near the container's limit.
_query_slots = threading.BoundedSemaphore(2)


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    with _query_slots:
        cur = con()
        try:
            return cur.execute(sql, params or []).df()
        finally:
            cur.close()


_frame_locks: dict[str, threading.Lock] = {}


def frame(table: str) -> pd.DataFrame:
    """A whole (small) table as a cached DataFrame, loaded once even if several threads ask at once."""
    with _lock:
        lock = _frame_locks.setdefault(table, threading.Lock())
    with lock:
        if table not in _frames:
            _frames[table] = query(f"SELECT * FROM {table}")  # table names are constants in this module
    return _frames[table]


# --------------------------------------------------------------------------------------------
# Comparisons shared by every source
# --------------------------------------------------------------------------------------------


def label(z: float) -> str:
    if np.isnan(z):
        return "can't tell"
    a = abs(z)
    if a >= CLEAR_Z:
        return "clear"
    if a >= SUGGESTIVE_Z:
        return "suggestive"
    return "no clear difference"


def comparisons(est: dict[int, float], var_of) -> list[str]:
    """Each weekday against the average of the other four, and the highest vs the lowest day.
    var_of(coefs) returns the variance of sum(coefs[d] * estimate[d])."""
    days = [d for d in WEEKDAYS if not np.isnan(est.get(d, np.nan))]
    if len(days) < 2:
        return []
    out = []
    for d in days:
        others = [k for k in days if k != d]
        coefs = {k: -1 / len(others) for k in others} | {d: 1.0}
        diff = sum(c * est[k] for k, c in coefs.items())
        se = np.sqrt(max(var_of(coefs), 0))
        z = diff / se if se > 0 else np.nan
        out.append(f"{WEEKDAYS[d]} vs other weekdays: {diff:+.2f} (SE {se:.2f}, z={z:+.1f}) -> {label(z)}")
    hi, lo = max(days, key=est.get), min(days, key=est.get)
    coefs = {hi: 1.0, lo: -1.0}
    diff, se = est[hi] - est[lo], np.sqrt(max(var_of(coefs), 0))
    z = diff / se if se > 0 else np.nan
    out.append(f"Highest ({WEEKDAYS[hi]}) vs lowest ({WEEKDAYS[lo]}): {diff:+.2f} (SE {se:.2f}, z={z:+.1f}) -> {label(z)}")
    return out


def table_text(title: str, est: dict, se: dict, n: dict, unit: str, min_n: int, var_of,
               thin_n: int = MIN_N, show_index: bool = True, n_label: str = "n") -> str:
    valid = {d: est[d] for d in WEEKDAYS if d in est and n.get(d, 0) >= min_n}
    mean = np.mean(list(valid.values())) if valid else np.nan
    lines = [f"{title}", f"  weekday | {unit} | SE | {n_label}" + (" | index (Mon-Fri avg = 100)" if show_index else "")]
    for d, name in WEEKDAYS.items():
        if d not in est:
            lines.append(f"  {name} | no data")
        elif n.get(d, 0) < min_n:
            lines.append(f"  {name} | too thin to report (n={n.get(d, 0)})")
        else:
            thin = " (thin)" if n[d] < thin_n else ""
            index = f" | {100 * est[d] / mean:.0f}" if show_index else ""
            lines.append(f"  {name} | {est[d]:.2f} | {se[d]:.2f} | {n[d]:,}{thin}{index}")
    if len(valid) >= 2:
        lines.append("  Comparisons:")
        lines += ["    " + c for c in comparisons(valid, var_of)]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# weekday_spending: CE Diary and DCPC (dollars)
# --------------------------------------------------------------------------------------------

CE_MEASURES = {"groceries": "food_home", "eating_out": "eating_out", "fast_food": "fast_food",
               "full_service": "full_service", "food_total": "food_home + eating_out",
               "lunch_out": "lunch_out", "dinner_out": "dinner_out", "breakfast_out": "breakfast_out",
               "snack_out": "snack_out"}
CE_SPLITS = {"none": "'all'", "period": "period", "year": "CAST(d.year AS VARCHAR)", "region": "h.region",
             "ny_metro": "CASE WHEN h.ny_metro THEN 'ny_metro' ELSE 'rest_of_us' END",
             "age_group": "h.age_group", "earners": "h.earners", "income_group": "h.income_group"}
DCPC_MEASURES = ["groceries", "eating_out", "fast_food", "full_service", "food_total"]
DCPC_SPLITS = ["none", "year", "employment", "age_group", "ny_state"]
REPS = [f"r{i}" for i in range(1, 45)]


def _years(start: int, end: int, lo: int, hi: int) -> tuple[int, int]:
    if start > end:
        start, end = end, start
    if end < lo or start > hi:
        raise ToolInputError(f"This source covers {lo}-{hi}; {start}-{end} is outside it.")
    return max(start, lo), min(end, hi)


def ce_spending(measure: str, start_year: int, end_year: int, split_by: str, geography: str) -> Result:
    if measure not in CE_MEASURES:
        raise ToolInputError(f"measure must be one of {list(CE_MEASURES)}")
    if split_by not in CE_SPLITS:
        raise ToolInputError(f"split_by must be one of {list(CE_SPLITS)} for ce_diary")
    if geography not in ("us", "ny_metro", "northeast"):
        raise ToolInputError("geography must be us, ny_metro or northeast for ce_diary")
    y0, y1 = _years(start_year, end_year, 2015, 2025)
    caveats = []
    if measure in ("lunch_out", "dinner_out", "breakfast_out", "snack_out"):
        if y0 > 2021:
            raise ToolInputError("Meal detail (lunch/dinner/breakfast/snack) exists only for 2015-2021.")
        y1 = min(y1, 2021)
        caveats.append("Meal detail exists only through 2021.")
    if y0 <= 2021 <= y1 and y1 >= 2022 and measure != "groceries":
        caveats.append("Food away from home was recoded in 2022; eating-out totals are comparable, meal detail is not.")
    if y0 <= 2020 <= y1:
        caveats.append("Includes 2020 (pandemic).")
    if y0 < 2022 and y1 - y0 >= 3:
        caveats.append("Dollars are not adjusted for inflation, so a multi-year average understates today's "
                       "prices (CE eating out per household per day rose from $7.06 in 2015 to $9.65 in 2025, "
                       "all days). For current levels query 2022-2025.")

    expr = CE_MEASURES[measure]
    group = CE_SPLITS[split_by]
    geo = {"us": "TRUE", "ny_metro": "h.ny_metro", "northeast": "h.region = 'northeast'"}[geography]
    rep_sums = ", ".join(f"SUM(h.{r} * ({expr})) AS s_{r}, SUM(h.{r}) AS w_{r}" for r in REPS)
    sql = f"""
        SELECT {group} AS grp, d.weekday, COUNT(*) AS n,
               SUM(h.w * ({expr})) AS s, SUM(h.w) AS wsum, {rep_sums}
        FROM (SELECT *, CASE WHEN year <= 2019 THEN '2015-2019' WHEN year <= 2021 THEN '2020-2021'
                             ELSE '2022-2025' END AS period FROM ce_day) d
        JOIN ce_hh h ON h.NEWID = d.NEWID AND h.year = d.year
        WHERE d.year BETWEEN ? AND ? AND d.weekday BETWEEN 2 AND 6 AND {geo}
        GROUP BY grp, d.weekday ORDER BY grp, d.weekday"""
    df = query(sql, [y0, y1])
    min_n = MIN_N_LOCAL if geography == "ny_metro" or split_by == "ny_metro" else MIN_N
    blocks = []
    for grp, g in df.groupby("grp"):
        g = g.set_index("weekday")
        est = (g.s / g.wsum).to_dict()
        reps = {r: (g[f"s_{r}"] / g[f"w_{r}"]).to_dict() for r in REPS}
        n = g.n.astype(int).to_dict()

        def var_of(coefs, est=est, reps=reps):
            base = sum(c * est[d] for d, c in coefs.items())
            v = sum((sum(c * reps[r][d] for d, c in coefs.items()) - base) ** 2 for r in REPS) / len(REPS)
            return v * SE_SCALE["ce_diary"] ** 2

        se = {d: np.sqrt(var_of({d: 1.0})) for d in est}
        title = f"[{split_by}={grp}]" if split_by != "none" else "[all households]"
        blocks.append(table_text(title, est, se, n, "$ per household per day", min_n, var_of))
    head = (f"CE Diary, {measure.replace('_', ' ')}, {y0}-{y1}, geography={geography}. Weighted (FINLWT21) mean "
            f"dollars per household per weekday; {SE_NOTE['ce_diary']}; n = diary-weeks (households).")
    text = head + "\n" + "\n".join(blocks) + ("\nCaveats: " + " ".join(caveats) if caveats else "")
    return Result(text, "ce_diary", {"measure": measure, "years": f"{y0}-{y1}", "split_by": split_by,
                                     "geography": geography}, caveats)


def dcpc_spending(measure: str, start_year: int, end_year: int, split_by: str) -> Result:
    if measure not in DCPC_MEASURES:
        raise ToolInputError(f"measure must be one of {DCPC_MEASURES} for dcpc")
    if split_by not in DCPC_SPLITS:
        raise ToolInputError(f"split_by must be one of {DCPC_SPLITS} for dcpc")
    y0, y1 = _years(start_year, end_year, 2021, 2025)
    df = frame("dcpc_day")
    df = df[df.year.between(y0, y1) & df.weekday.between(2, 6)]
    caveats = ["October diaries only; 'groceries' includes pharmacies and convenience stores.",
               "DCPC is small: in a split-half test its weekday differences replicated only ~70% of the time, "
               "so treat DCPC-only results as suggestive unless the CE Diary agrees."]
    if split_by == "ny_state":
        df = df[df.year >= 2022]
        caveats.append("State is recorded from 2022; New York State, not NYC.")
    groups = df.assign(grp="all") if split_by == "none" else df.assign(
        grp=df[split_by].map(lambda v: {True: "new_york_state", False: "other_states"}.get(v, v)))
    min_n = MIN_N_LOCAL if split_by == "ny_state" else MIN_N
    blocks = []
    for grp, g in groups.groupby("grp"):
        y = g[measure].to_numpy()
        w = g.w.to_numpy()
        est, se, n, infl = {}, {}, {}, {}
        for d in WEEKDAYS:
            m = (g.weekday == d).to_numpy()
            if m.sum() == 0:
                continue
            W = w[m].sum()
            est[d] = float((w[m] * y[m]).sum() / W)
            n[d] = int(m.sum())
            e = np.zeros(len(g))
            e[m] = w[m] * (y[m] - est[d]) / W
            infl[d] = pd.Series(e).groupby(g.person.to_numpy()).sum()
        persons = len(g.person.unique())

        def var_of(coefs, infl=infl, persons=persons):
            tot = sum(c * infl[d] for d, c in coefs.items())
            return float((tot ** 2).sum() * persons / max(persons - 1, 1))

        for d in est:
            se[d] = np.sqrt(var_of({d: 1.0}))
        title = f"[{split_by}={grp}]" if split_by != "none" else "[all people]"
        blocks.append(table_text(title, est, se, n, "$ per person per day", min_n, var_of))
    head = (f"DCPC, {measure.replace('_', ' ')}, {y0}-{y1}. Weighted (daily_weight) mean dollars per person per "
            f"diary day, each measure capped at the year's 99.5th percentile; SEs clustered by person; n = person-days.")
    return Result(head + "\n" + "\n".join(blocks) + "\nCaveats: " + " ".join(caveats), "dcpc",
                  {"measure": measure, "years": f"{y0}-{y1}", "split_by": split_by}, caveats)


# --------------------------------------------------------------------------------------------
# weekday_eating: NHANES (precomputed)
# --------------------------------------------------------------------------------------------

NHANES_MEASURES = ["pct_kcal_restaurant", "pct_kcal_full_service", "pct_kcal_fast_food", "pct_kcal_store",
                   "pct_kcal_workplace_cafeteria", "pct_ate_restaurant", "pct_ate_restaurant_lunch",
                   "pct_ate_restaurant_dinner"]
NHANES_GROUPS = ["all", "children_2_19", "adults_20_64", "employed_20_64", "not_employed_20_64", "age_65_plus"]


def nhanes_eating(measure: str, cycle: str, groups: list[str]) -> Result:
    if measure not in NHANES_MEASURES:
        raise ToolInputError(f"measure must be one of {NHANES_MEASURES}")
    cycles = ["2017-2018", "2021-2023"] if cycle == "both" else [cycle]
    if any(c not in ("2017-2018", "2021-2023") for c in cycles):
        raise ToolInputError("cycle must be 2017-2018, 2021-2023 or both")
    groups = groups or ["all"]
    bad = [g for g in groups if g not in NHANES_GROUPS]
    if bad:
        raise ToolInputError(f"unknown groups {bad}; choose from {NHANES_GROUPS}")
    t = frame("nhanes_weekday")
    blocks = []
    for c in cycles:
        for gname in groups[:4]:
            g = t[(t.cycle == c) & (t.group == gname) & (t.measure == measure)].set_index("weekday")
            est, se, n = g.estimate.to_dict(), g.se.to_dict(), g.n.astype(int).to_dict()

            def var_of(coefs, se=se):  # different people on each weekday: treat days as independent
                return sum(cf ** 2 * se[d] ** 2 for d, cf in coefs.items())

            blocks.append(table_text(f"[cycle={c}, group={gname}]", est, se, n, g.unit.iloc[0] if len(g) else "%",
                                     MIN_N, var_of))
    caveats = ["No dollars and no geography.", "2021-2023 recalls were by phone; 2017-2018 in person."]
    head = (f"NHANES, {measure}. Design-weighted (WTDRD1) estimates by intake weekday with Taylor-linearized SEs; "
            f"n = people.")
    return Result(head + "\n" + "\n".join(blocks) + "\nCaveats: " + " ".join(caveats), "nhanes",
                  {"measure": measure, "cycle": cycle, "groups": groups[:4]}, caveats)


# --------------------------------------------------------------------------------------------
# weekday_time_use: ATUS
# --------------------------------------------------------------------------------------------

ATUS_ACTIVITIES = ["grocery", "buy_prepared_food", "eat_restaurant", "eat_at_workplace", "eat_home", "food_prep"]
ATUS_SPLITS = ["none", "work_location", "usual_teleworker", "age_group", "period", "year"]


def atus_time_use(activity: str, stat: str, start_year: int, end_year: int, split_by: str,
                  geography: str, employed_only: bool) -> Result:
    if activity not in ATUS_ACTIVITIES:
        raise ToolInputError(f"activity must be one of {ATUS_ACTIVITIES}")
    if stat not in ("share", "minutes"):
        raise ToolInputError("stat must be share (percent who did it that day) or minutes (average minutes)")
    if split_by not in ATUS_SPLITS:
        raise ToolInputError(f"split_by must be one of {ATUS_SPLITS}")
    if geography not in ("us", "ny_metro", "nyc"):
        raise ToolInputError("geography must be us, ny_metro or nyc")
    y0, y1 = _years(start_year, end_year, 2015, 2025)
    df = frame("atus_day")
    df = df[df.year.between(y0, y1) & df.weekday.between(2, 6) & (df.holiday != 1)]
    if geography != "us":
        df = df[df[geography]]
    if employed_only:
        df = df[df.employed]
    col = f"{activity}_any" if stat == "share" else f"{activity}_min"
    scale = 100.0 if stat == "share" else 1.0
    caveats = ["No dollars."]
    if y0 <= 2020 <= y1:
        caveats.append("2020 has no diaries from mid-March to mid-May.")
    if y1 >= 2025:
        caveats.append("October 2025 is missing (federal shutdown).")
    if split_by == "usual_teleworker":
        df = df[df.usual_teleworker != "unknown"]
        caveats.append("Usual telework comes from the CPS interview 2-5 months before the diary day; recorded mainly 2023+.")
    if split_by == "period":
        df = df.assign(period=np.where(df.year <= 2019, "2015-2019", np.where(df.year <= 2021, "2020-2021", "2022-2025")))
    grp_col = None if split_by == "none" else split_by
    min_n = MIN_N_LOCAL if geography != "us" else MIN_N
    blocks = []
    for grp, g in (df.groupby(grp_col) if grp_col else [("all", df)]):
        est, se, n = {}, {}, {}
        for d in WEEKDAYS:
            s = g[g.weekday == d]
            if len(s) == 0:
                continue
            w, y = s.w.to_numpy(), scale * s[col].to_numpy()
            m = (w * y).sum() / w.sum()
            est[d], n[d] = float(m), len(s)
            se[d] = float(np.sqrt((w ** 2 * (y - m) ** 2).sum()) / w.sum() * np.sqrt(len(s) / max(len(s) - 1, 1)))

        se = {d: v * SE_SCALE["atus"] for d, v in se.items()}

        def var_of(coefs, se=se):  # one diary day per person: weekdays are independent samples
            return sum(c ** 2 * se[d] ** 2 for d, c in coefs.items())

        unit = "% who did it" if stat == "share" else "avg minutes"
        blocks.append(table_text(f"[{split_by}={grp}]" if grp_col else "[all respondents]", est, se, n, unit, min_n, var_of))
    who = "employed people" if employed_only else "everyone 15+"
    head = (f"ATUS, {activity.replace('_', ' ')} ({stat}), {y0}-{y1}, {who}, geography={geography}, holidays excluded. "
            f"Weighted (TUFNWGTP; TU20FWGT for 2020); {SE_NOTE['atus']}; n = respondents (one diary day each).")
    return Result(head + "\n" + "\n".join(blocks) + "\nCaveats: " + " ".join(caveats), "atus",
                  {"activity": activity, "stat": stat, "years": f"{y0}-{y1}", "split_by": split_by,
                   "geography": geography, "employed_only": employed_only}, caveats)


# --------------------------------------------------------------------------------------------
# payday_window: food on common paydays vs other days of the month (ATUS all year, DCPC October)
# --------------------------------------------------------------------------------------------

PAY_WINDOWS = {"1st-3rd": (1, 3), "15th-17th": (15, 17), "28th-31st": (28, 31)}


def payday_window(source: str, measure: str, stat: str, start_year: int, end_year: int) -> Result:
    """No survey records anyone's payday, so this compares days that often follow one (monthly and
    semimonthly pay, Social Security on the 3rd) with the rest of the month. Each window is compared
    with 'other days' within the same weekday and then averaged over Mon-Fri, so the weekday mix of each
    window can't create a difference."""
    if source == "atus":
        if measure not in ATUS_ACTIVITIES:
            raise ToolInputError(f"activity must be one of {ATUS_ACTIVITIES}")
        if stat not in ("share", "minutes"):
            raise ToolInputError("stat must be share or minutes")
        y0, y1 = _years(start_year, end_year, 2015, 2025)
        df = frame("atus_day")
        df = df[df.year.between(y0, y1) & df.weekday.between(2, 6) & (df.holiday != 1)]
        col = f"{measure}_any" if stat == "share" else f"{measure}_min"
        y = df[col].to_numpy(float) * (100.0 if stat == "share" else 1.0)
        cluster = df.TUCASEID.to_numpy()
        unit = "% who did it" if stat == "share" else "avg minutes"
        head = (f"ATUS, {measure.replace('_', ' ')} ({stat}), {y0}-{y1}, everyone 15+, weekdays, holidays excluded, "
                f"all months.")
        caveats = ["No dollars."]
    elif source == "dcpc":
        if measure not in DCPC_MEASURES:
            raise ToolInputError(f"measure must be one of {DCPC_MEASURES}")
        y0, y1 = _years(start_year, end_year, 2021, 2025)
        df = frame("dcpc_day")
        df = df[df.year.between(y0, y1) & df.weekday.between(2, 6)]
        y = df[measure].to_numpy(float)
        cluster = df.person.to_numpy()
        unit = "$ per person per day"
        head = f"DCPC, {measure.replace('_', ' ')}, October {y0}-{y1}, weekdays."
        caveats = ["October only.", "DCPC is small; treat DCPC-only results as suggestive unless another source agrees."]
    else:
        raise ToolInputError("source must be atus or dcpc")
    dom = pd.to_datetime(df.date.astype(str), format="mixed").dt.day.to_numpy()
    window = np.full(len(df), "other", dtype=object)
    for name, (a, b) in PAY_WINDOWS.items():
        window[(dom >= a) & (dom <= b)] = name
    w, wd = df.w.to_numpy(float), df.weekday.to_numpy()
    days = [d for d in WEEKDAYS if (wd == d).any()]
    cells, infl = {}, {}
    for name in list(PAY_WINDOWS) + ["other"]:
        for d in days:
            m = (window == name) & (wd == d)
            if m.sum() == 0:
                continue
            W = w[m].sum()
            cells[(name, d)] = (w[m] * y[m]).sum() / W
            e = np.zeros(len(df))
            e[m] = w[m] * (y[m] - cells[(name, d)]) / W
            infl[(name, d)] = e
    k = len(np.unique(cluster))
    scale = SE_SCALE.get(source, 1.0)
    lines = [f"  window | {unit} (weekday-adjusted) | vs other days | SE | z | label | n"]
    other = np.mean([cells[("other", d)] for d in days])
    for name in PAY_WINDOWS:
        ok = [d for d in days if (name, d) in cells and ("other", d) in cells]
        if len(ok) < len(days):
            lines.append(f"  {name} | too few days")
            continue
        level = np.mean([cells[(name, d)] for d in ok])
        diff = np.mean([cells[(name, d)] - cells[("other", d)] for d in ok])
        tot = sum(infl[(name, d)] - infl[("other", d)] for d in ok) / len(ok)
        se = np.sqrt((pd.Series(tot).groupby(cluster).sum() ** 2).sum() * k / max(k - 1, 1)) * scale
        z = diff / se if se > 0 else np.nan
        n = int((window == name).sum())
        lines.append(f"  {name} | {level:.2f} | {diff:+.2f} | {se:.2f} | {z:+.1f} | {label(z)} | {n:,}")
    lines.append(f"  other days | {other:.2f} | | | | | {int((window == 'other').sum()):,}")
    caveats += ["Nobody's actual payday is recorded: these are days that often follow one.",
                "Every-other-Friday paydays can't be separated from Friday itself."]
    text = (head + " Each window is compared with other days of the month on the same weekday, then averaged "
            "over Mon-Fri.\n" + "\n".join(lines) + "\nCaveats: " + " ".join(caveats))
    return Result(text, source, {"measure": measure, **({"stat": stat} if source == "atus" else {}),
                                 "years": f"{y0}-{y1}", "test": "payday windows"}, caveats)


# --------------------------------------------------------------------------------------------
# nyc_ridership: live MTA data (see mta.py for its guardrails)
# --------------------------------------------------------------------------------------------


def nyc_ridership(year: int, station_set: str, season: str, hours: str) -> Result:
    import mta

    try:
        r = mta.weekday_shares(year, station_set, season, hours)
    except ValueError as e:
        raise ToolInputError(str(e))
    keys = {2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri"}
    avg = r["weekday_avg"]
    est = {d: r["index"][k] for d, k in keys.items()}
    se = {d: 100 * r["se_entries"][k] / avg for d, k in keys.items()}
    n = {d: r["days_used"][k] for d, k in keys.items()}

    def var_of(coefs, se=se):  # different calendar days: independent
        return sum(c ** 2 * se[d] ** 2 for d, c in coefs.items())

    where = mta.STATION_NAMES if station_set == "office_11" else "all subway stations"
    title = f"[{station_set}: {where}]"
    body = table_text(title, est, se, n, "index of entries (Mon-Fri avg = 100)", 3, var_of,
                      thin_n=0, show_index=False, n_label="days")
    head = (f"MTA subway entries, {r['season']} {r['year']} ({r['first_day']} to {r['last_day']}), hours={r['hours']}, "
            f"federal holidays excluded. Mon-Fri average {avg:,.0f} entries per day. SEs from day-to-day variation.")
    caveats = ["Paid entries only (fare evasion not counted); no food or spending information."]
    return Result(head + "\n" + body + "\nCaveats: " + caveats[0], "mta",
                  {"year": r["year"], "station_set": station_set, "season": season, "hours": hours}, caveats)


def reference_note(topic: str) -> Result:
    from catalog import REFERENCE_NOTES

    if topic not in REFERENCE_NOTES:
        raise ToolInputError(f"topic must be one of {list(REFERENCE_NOTES)}")
    return Result(REFERENCE_NOTES[topic], "reference", {"topic": topic},
                  ["Published figures quoted by hand, not data the agent computed."])
