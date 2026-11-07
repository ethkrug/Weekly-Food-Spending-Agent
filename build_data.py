"""Offline build: turn the raw survey files in `local data/` into one small DuckDB file.

The app never reads raw survey files and the model never does survey math. Everything that is
easy to get wrong -- weights, which purchases count as "eating out", which day a diary entry
belongs to, design-based standard errors -- happens here, once, in code.

    uv run --group build build_data.py                     # reads ../local data
    uv run --group build build_data.py --raw /path/to/raw  # or somewhere else

Writes data/food_week.duckdb with these tables:

    ce_hh, ce_day        BLS Consumer Expenditure Diary 2015-2025: one row per diary week
                         (household), and one row per diary week x weekday with food dollars
    dcpc_day             Atlanta Fed Diary of Consumer Payment Choice 2021-2025: one row per
                         person x diary day with food dollars by merchant type
    nhanes_weekday       CDC NHANES 2017-18 and 2021-23: precomputed design-weighted estimates by
                         weekday x group x measure, with standard errors
    atus_day             BLS American Time Use Survey 2015-2025: one row per respondent (one
                         diary day each) with work location that day and food activities
    build_info           when and from what the file was built
"""

import argparse
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DEFAULT_RAW = HERE.parent / "local data"
OUT = HERE / "data" / "food_week.duckdb"

WEEKDAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
NYC_COUNTIES = {5, 47, 61, 81, 85}  # Bronx, Kings, New York, Queens, Richmond (state FIPS 36)
NY_METRO_CBSA = 35620  # New York-Newark-Jersey City, NY-NJ-PA


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def age_group(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 34, 54, 64, 200], labels=["under_35", "35_54", "55_64", "65_plus"]).astype(str)


# --------------------------------------------------------------------------------------------
# BLS Consumer Expenditure Diary
# --------------------------------------------------------------------------------------------

FAST_FOOD = {"190111", "190211", "190311", "190321", "190400"}
FULL_SERVICE = {"190112", "190212", "190312", "190322", "190500"}
VENDING_EMPLOYER = {"190113", "190213", "190313", "190323", "190114", "190214", "190314", "190324", "190600", "190700"}
# Meal detail exists only through the 2021 release; from 2022 food away is coded by outlet only.
MEAL = {"1901": "lunch", "1902": "dinner", "19031": "snack", "19032": "breakfast"}


def parse_stub(path: Path) -> dict[str, set[str]]:
    """UCC -> set of group codes above it, from a CE hierarchical grouping ('stub') file."""
    ancestors: dict[str, set[str]] = {}
    stack: list[tuple[int, str]] = []
    for line in path.read_text(encoding="latin1").splitlines():
        m = re.match(r"^1\s+(\d+)\s+(.*?)\s{2,}(\S+)\s+([GDSIT])\s", line)
        if not m:
            continue
        level, code = int(m.group(1)), m.group(3)
        while stack and stack[-1][0] >= level:
            stack.pop()
        if m.group(4) == "D" and code.isdigit():
            # The same UCC appears again later in the gifts section: keep the main hierarchy.
            ancestors.setdefault(code, {c for _, c in stack})
        stack.append((level, code))
    return ancestors


def ce_category_map(raw: Path, year: int) -> dict[str, str]:
    stubs = raw / "stubs"
    path = stubs / f"CE-HG-Diary-{year}.txt"
    if not path.exists():  # no 2025 diary stub yet: the 2022+ food codes haven't changed
        path = max(stubs.glob("CE-HG-Diary-20*.txt"))
    out = {}
    for ucc, groups in parse_stub(path).items():
        if groups & {"FOODHOME", "FOODHO"}:  # older stubs truncate group codes to 6 characters
            out[ucc] = "food_home"
        elif groups & {"FOODAWAY", "FOODAW"}:
            out[ucc] = "food_away"
    return out


def build_ce(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hh_frames, day_frames = [], []
    for year in range(2015, 2026):
        folder = raw / f"diary{year % 100:02d}"
        if not folder.exists():
            log(f"CE {year}: missing, skipped")
            continue
        cats = ce_category_map(raw, year)
        for q in range(1, 5):
            fmld = pd.read_sas(folder / f"fmld{year % 100:02d}{q}.sas7bdat", encoding="latin1")
            expd = pd.read_sas(folder / f"expd{year % 100:02d}{q}.sas7bdat", encoding="latin1")
            wt = ["FINLWT21"] + [f"WTREP{i:02d}" for i in range(1, 45)]
            hh = fmld[["NEWID", "REGION", "STATE", "PSU", "AGE_REF", "EARNCOMP", "FAM_SIZE", "INC_RANK"] + wt].copy()
            hh["year"], hh["quarter"] = year, q
            hh_frames.append(hh)

            e = expd[["NEWID", "UCC", "COST", "EXPNWKDY"]].copy()
            e["UCC"] = e.UCC.astype(str).str.strip()
            e["cat"] = e.UCC.map(cats)
            e = e[e.cat.notna() & e.EXPNWKDY.astype(str).str.strip().isin([str(i) for i in range(1, 8)])]
            e["weekday"] = e.EXPNWKDY.astype(int)
            e["food_home"] = np.where(e.cat == "food_home", e.COST, 0.0)
            away = e.cat == "food_away"
            e["fast_food"] = np.where(away & e.UCC.isin(FAST_FOOD), e.COST, 0.0)
            e["full_service"] = np.where(away & e.UCC.isin(FULL_SERVICE), e.COST, 0.0)
            e["vending_employer"] = np.where(away & e.UCC.isin(VENDING_EMPLOYER), e.COST, 0.0)
            e["eating_out"] = e.fast_food + e.full_service + e.vending_employer  # excludes catered affairs, school board
            for prefix, meal in MEAL.items():
                e[f"{meal}_out"] = np.where(away & e.UCC.str.startswith(prefix) & (year <= 2021), e.COST, 0.0)
            cols = ["food_home", "fast_food", "full_service", "vending_employer", "eating_out",
                    "lunch_out", "dinner_out", "snack_out", "breakfast_out"]
            sums = e.groupby(["NEWID", "weekday"])[cols].sum()
            # Every diary week covers each weekday exactly once. Days with no food purchase are zeros.
            grid = pd.MultiIndex.from_product([fmld.NEWID.unique(), range(1, 8)], names=["NEWID", "weekday"])
            day = sums.reindex(grid, fill_value=0.0).reset_index()
            day["year"] = year
            day_frames.append(day)
        log(f"CE {year}: done")
    hh = pd.concat(hh_frames, ignore_index=True)
    hh["ny_metro"] = hh.PSU.astype(str).isin(["S12A", "1109", "1110", "1111"])
    hh["state"] = pd.to_numeric(hh.STATE, errors="coerce")
    hh["region"] = pd.to_numeric(hh.REGION, errors="coerce").map({1: "northeast", 2: "midwest", 3: "south", 4: "west"})
    hh["age_group"] = age_group(pd.to_numeric(hh.AGE_REF, errors="coerce"))
    hh["earners"] = np.where(pd.to_numeric(hh.EARNCOMP, errors="coerce") == 8, "no_earners", "has_earners")
    hh["income_group"] = pd.cut(pd.to_numeric(hh.INC_RANK, errors="coerce"), [-0.01, 0.333, 0.667, 1.01],
                                labels=["low", "middle", "high"]).astype(str)
    hh = hh.rename(columns={"FINLWT21": "w"} | {f"WTREP{i:02d}": f"r{i}" for i in range(1, 45)})
    # A household's two diary weeks share an NEWID prefix but are separate weighted units.
    hh = hh.drop(columns=["REGION", "STATE", "AGE_REF", "EARNCOMP", "INC_RANK"])
    day = pd.concat(day_frames, ignore_index=True)
    # Same NEWID can recur across years; key by (year, NEWID).
    return hh, day


# --------------------------------------------------------------------------------------------
# Atlanta Fed Diary of Consumer Payment Choice
# --------------------------------------------------------------------------------------------


def build_dcpc(raw: Path) -> pd.DataFrame:
    frames = []
    for year in range(2021, 2026):
        files = list((raw / "dcpc" / str(year)).glob("*.csv"))
        tran = next(f for f in files if "tran" in f.name)
        dayf = next(f for f in files if "day" in f.name)
        ind = next(f for f in files if "ind" in f.name)
        t = pd.read_csv(tran, low_memory=False, usecols=lambda c: c in ("id", "date", "merch", "amnt", "diary_day"))
        d = pd.read_csv(dayf, low_memory=False, usecols=lambda c: c in ("id", "date", "diary_day", "daily_weight"))
        i = pd.read_csv(ind, low_memory=False, usecols=lambda c: c in (
            "id", "age", "work_employed", "work_retired", "workfullpart", "statereside", "urban_cat"))
        d = d[d.diary_day.isin([1, 2, 3]) & (d.daily_weight > 0)].rename(columns={"daily_weight": "w"})
        t = t[t.merch.isin([1, 3, 4]) & t.diary_day.isin([1, 2, 3]) & (t.amnt > 0)]
        s = t.groupby(["id", "date", "merch"]).amnt.sum().unstack(fill_value=0).reindex(columns=[1, 3, 4], fill_value=0)
        s.columns = ["groceries", "full_service", "fast_food"]
        x = d.merge(s.reset_index(), on=["id", "date"], how="left").fillna({"groceries": 0, "full_service": 0, "fast_food": 0})
        x["eating_out"] = x.full_service + x.fast_food
        x["food_total"] = x.groceries + x.eating_out
        # Diary dollars have a long right tail (a $900 catering order). Cap each measure at the
        # year's 99.5th percentile so one entry can't move a weekday average.
        for c in ["groceries", "full_service", "fast_food", "eating_out", "food_total"]:
            x[c] = x[c].clip(upper=x[c].quantile(0.995))
        x["year"] = year
        x["weekday"] = pd.to_datetime(x.date).dt.dayofweek.map({6: 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7})
        x = x.merge(i, on="id", how="left")
        x["employment"] = np.select(
            [(x.work_employed == 1) & (x.workfullpart == 1), x.work_employed == 1, x.work_retired == 1],
            ["full_time", "part_time", "retired"], "not_employed")
        x["age_group"] = age_group(x.age)
        x["ny_state"] = x.get("statereside") == 32  # NY = 32 in the 2022-2025 codebooks; absent in 2021
        x["person"] = x.year.astype(str) + "_" + x.id.astype(str)
        frames.append(x[["person", "year", "date", "weekday", "w", "groceries", "full_service", "fast_food",
                         "eating_out", "food_total", "employment", "age_group", "ny_state"]])
        log(f"DCPC {year}: {len(x):,} person-days")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------------
# CDC NHANES What We Eat in America
# --------------------------------------------------------------------------------------------


def _xpt(path: Path, cols: list[str]) -> pd.DataFrame:
    d = pd.read_sas(path, format="xport")
    d = d[[c for c in cols if c in d.columns]].copy()
    # read_sas returns exact zeros as 5.4e-79: round them back
    for c in d.select_dtypes("number").columns:
        d.loc[d[c].abs() < 1e-30, c] = 0.0
    return d


def ratio_se(df: pd.DataFrame, num: str, den: str, domain: pd.Series) -> tuple[float, float]:
    """Design-based (Taylor linearization) estimate and SE of sum(w*num)/sum(w*den) over a domain,
    using NHANES masked strata/PSUs. Rows outside the domain contribute zero."""
    w = df.w * domain
    tot_den = (w * df[den]).sum()
    if tot_den <= 0:
        return np.nan, np.nan
    r = (w * df[num]).sum() / tot_den
    z = w * (df[num] - r * df[den]) / tot_den
    psu = pd.DataFrame({"h": df.strata, "j": df.psu, "z": z}).groupby(["h", "j"]).z.sum().reset_index()
    var = 0.0
    for _, g in psu.groupby("h"):
        n = len(g)
        if n > 1:
            var += n / (n - 1) * ((g.z - g.z.mean()) ** 2).sum()
    return r, float(np.sqrt(var))


def build_nhanes(raw: Path) -> pd.DataFrame:
    rows = []
    for cycle, sfx in [("2017-2018", "J"), ("2021-2023", "L")]:
        iff = _xpt(raw / "nhanes" / f"DR1IFF_{sfx}.xpt", ["SEQN", "DR1FS", "DR1IKCAL", "DR1_020"])
        tot = _xpt(raw / "nhanes" / f"DR1TOT_{sfx}.xpt", ["SEQN", "DR1DAY", "DR1DRSTZ", "WTDRD1"])
        demo = _xpt(raw / "nhanes" / f"DEMO_{sfx}.xpt", ["SEQN", "RIDAGEYR", "SDMVPSU", "SDMVSTRA"])
        ocq = _xpt(raw / "nhanes" / f"OCQ_{sfx}.xpt", ["SEQN", "OCD150"])
        fs = iff.DR1FS
        restaurant = fs.isin([2, 3, 4, 5])
        lunch = iff.DR1_020.between(10.5 * 3600, 14 * 3600)
        dinner = iff.DR1_020.between(17 * 3600, 21 * 3600)
        iff = iff.assign(
            kcal=iff.DR1IKCAL,
            kcal_restaurant=np.where(restaurant, iff.DR1IKCAL, 0),
            kcal_full_service=np.where(fs == 2, iff.DR1IKCAL, 0),
            kcal_fast_food=np.where(fs == 3, iff.DR1IKCAL, 0),
            kcal_store=np.where(fs.isin([1, 27, 28]), iff.DR1IKCAL, 0),
            kcal_cafeteria=np.where(fs == 6, iff.DR1IKCAL, 0),
            any_restaurant=restaurant.astype(float),
            restaurant_lunch=(restaurant & lunch).astype(float),
            restaurant_dinner=(restaurant & dinner).astype(float),
        )
        p = iff.groupby("SEQN").agg(
            kcal=("kcal", "sum"), kcal_restaurant=("kcal_restaurant", "sum"), kcal_full_service=("kcal_full_service", "sum"),
            kcal_fast_food=("kcal_fast_food", "sum"), kcal_store=("kcal_store", "sum"), kcal_cafeteria=("kcal_cafeteria", "sum"),
            ate_restaurant=("any_restaurant", "max"), ate_restaurant_lunch=("restaurant_lunch", "max"),
            ate_restaurant_dinner=("restaurant_dinner", "max"),
        ).reset_index()
        p = p.merge(tot, on="SEQN").merge(demo, on="SEQN").merge(ocq, on="SEQN", how="left")
        p = p[(p.DR1DRSTZ == 1) & (p.WTDRD1 > 1) & (p.RIDAGEYR >= 2) & (p.kcal > 0)].copy()
        p = p.rename(columns={"WTDRD1": "w", "SDMVSTRA": "strata", "SDMVPSU": "psu"})
        p["one"] = 1.0
        age = p.RIDAGEYR
        employed = p.OCD150.isin([1, 2])
        groups = {
            "all": age >= 2,
            "children_2_19": age < 20,
            "adults_20_64": age.between(20, 64),
            "employed_20_64": age.between(20, 64) & employed,
            "not_employed_20_64": age.between(20, 64) & ~employed,
            "age_65_plus": age >= 65,
        }
        measures = {  # name -> (numerator, denominator, unit)
            "pct_kcal_restaurant": ("kcal_restaurant", "kcal", "percent of calories"),
            "pct_kcal_full_service": ("kcal_full_service", "kcal", "percent of calories"),
            "pct_kcal_fast_food": ("kcal_fast_food", "kcal", "percent of calories"),
            "pct_kcal_store": ("kcal_store", "kcal", "percent of calories"),
            "pct_kcal_workplace_cafeteria": ("kcal_cafeteria", "kcal", "percent of calories"),
            "pct_ate_restaurant": ("ate_restaurant", "one", "percent of people"),
            "pct_ate_restaurant_lunch": ("ate_restaurant_lunch", "one", "percent of people"),
            "pct_ate_restaurant_dinner": ("ate_restaurant_dinner", "one", "percent of people"),
        }
        for gname, gmask in groups.items():
            for day in range(2, 7):  # Monday..Friday
                dom = (gmask & (p.DR1DAY == day)).astype(float)
                n = int(dom.sum())
                for mname, (num, den, unit) in measures.items():
                    est, se = ratio_se(p, num, den, dom)
                    rows.append({"cycle": cycle, "group": gname, "weekday": day, "measure": mname, "unit": unit,
                                 "estimate": 100 * est, "se": 100 * se, "n": n})
        log(f"NHANES {cycle}: {len(p):,} people")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# BLS American Time Use Survey
# --------------------------------------------------------------------------------------------


def build_atus(raw: Path) -> pd.DataFrame:
    resp = pd.read_csv(raw / "atusresp-0325" / "atusresp_0325.dat", usecols=[
        "TUCASEID", "TUYEAR", "TUDIARYDAY", "TUDIARYDATE", "TUFNWGTP", "TU20FWGT", "TRHOLIDAY", "TELFS", "TRDPFTPT"])
    resp = resp[resp.TUYEAR >= 2015]
    cps = pd.read_csv(raw / "atuscps-0325" / "atuscps_0325.dat", usecols=[
        "TUCASEID", "TULINENO", "GESTFIPS", "GTCBSA", "GTCO", "PRTAGE", "PTTLWK"])
    cps = cps[cps.TULINENO == 1].drop(columns="TULINENO")
    log("ATUS: reading activity file (large)")
    act = pd.read_csv(raw / "atusact-0325" / "atusact_0325.dat",
                      usecols=["TUCASEID", "TRCODEP", "TEWHERE", "TUACTDUR24"])
    act = act[act.TUCASEID.isin(resp.TUCASEID)]
    code = act.TRCODEP.astype(int)
    work = code.isin([50101, 50102])
    dur = act.TUACTDUR24
    act = act.assign(
        work_home=np.where(work & (act.TEWHERE == 1), dur, 0),
        work_workplace=np.where(work & (act.TEWHERE == 2), dur, 0),
        work_other=np.where(work & ~act.TEWHERE.isin([1, 2]), dur, 0),
        grocery_min=np.where(code == 70101, dur, 0),
        buy_prepared_food_min=np.where(code == 70103, dur, 0),
        eat_restaurant_min=np.where((code == 110101) & (act.TEWHERE == 4), dur, 0),
        eat_at_workplace_min=np.where((code == 110101) & (act.TEWHERE == 2), dur, 0),
        eat_home_min=np.where((code == 110101) & (act.TEWHERE == 1), dur, 0),
        food_prep_min=np.where((code >= 20201) & (code <= 20299), dur, 0),
    )
    cols = ["work_home", "work_workplace", "work_other", "grocery_min", "buy_prepared_food_min",
            "eat_restaurant_min", "eat_at_workplace_min", "eat_home_min", "food_prep_min"]
    p = act.groupby("TUCASEID")[cols].sum().reset_index()
    p = resp.merge(p, on="TUCASEID", how="left").merge(cps, on="TUCASEID", how="left").fillna({c: 0 for c in cols})
    p["w"] = np.where(p.TUYEAR == 2020, p.TU20FWGT, p.TUFNWGTP)
    p = p[p.w > 0]
    home, office, other = p.work_home >= 60, p.work_workplace >= 60, p.work_other >= 60
    p["work_location"] = np.select(
        [home & office, office, home, other], ["mixed", "office", "home", "elsewhere"], "no_work")
    for c in ["grocery_min", "buy_prepared_food_min", "eat_restaurant_min", "eat_at_workplace_min", "eat_home_min", "food_prep_min"]:
        p[c.replace("_min", "_any")] = (p[c] > 0).astype(float)
    p["employed"] = p.TELFS.isin([1, 2])
    p["usual_teleworker"] = np.where(p.PTTLWK == 1, "yes", np.where(p.PTTLWK == 2, "no", "unknown"))
    p["age_group"] = age_group(p.PRTAGE)
    p["ny_metro"] = p.GTCBSA == NY_METRO_CBSA
    p["nyc"] = (p.GESTFIPS == 36) & p.GTCO.isin(NYC_COUNTIES)
    p = p.rename(columns={"TUYEAR": "year", "TUDIARYDAY": "weekday", "TUDIARYDATE": "date", "TRHOLIDAY": "holiday"})
    keep = ["TUCASEID", "year", "date", "weekday", "holiday", "w", "work_location", "employed", "usual_teleworker",
            "age_group", "ny_metro", "nyc"] + cols + [c.replace("_min", "_any") for c in cols if c.endswith("_min")]
    log(f"ATUS: {len(p):,} respondents 2015-2025")
    return p[keep]


# --------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--fresh", action="store_true", help="ignore cached intermediate tables")
    args = ap.parse_args()
    raw = args.raw
    cache = OUT.parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    def cached(name, fn):
        # Reading 88 SAS files takes minutes: keep each source's table so a rebuild only redoes what changed.
        path = cache / f"{name}.parquet"
        if path.exists() and not args.fresh:
            return pd.read_parquet(path)
        df = fn()
        df.to_parquet(path, index=False)
        return df
    OUT.parent.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))

    ce = {}
    ce_hh = cached("ce_hh", lambda: ce.setdefault("t", build_ce(raw))[0])
    ce_day = cached("ce_day", lambda: ce.setdefault("t", build_ce(raw))[1])
    con.execute("CREATE TABLE ce_hh AS SELECT * FROM ce_hh")
    con.execute("CREATE TABLE ce_day AS SELECT * FROM ce_day")
    dcpc = cached("dcpc_day", lambda: build_dcpc(raw))
    con.execute("CREATE TABLE dcpc_day AS SELECT * FROM dcpc")
    nhanes = cached("nhanes_weekday", lambda: build_nhanes(raw))
    con.execute("CREATE TABLE nhanes_weekday AS SELECT * FROM nhanes")
    atus = cached("atus_day", lambda: build_atus(raw))
    con.execute("CREATE TABLE atus_day AS SELECT * FROM atus")

    info = pd.DataFrame([{"built_at": time.strftime("%Y-%m-%d %H:%M"), "raw": str(raw),
                          "ce_diary_weeks": len(ce_hh), "dcpc_person_days": len(dcpc),
                          "nhanes_rows": len(nhanes), "atus_respondents": len(atus)}])
    con.execute("CREATE TABLE build_info AS SELECT * FROM info")
    con.close()
    tmp.replace(OUT)
    log(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
