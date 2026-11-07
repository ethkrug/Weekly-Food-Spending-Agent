"""Test the 'is this weekday difference real?' rule against the data itself.

Two checks, for every comparison the tools make ("this weekday vs the average of the other four"):

1. Replication. Split each source into two independent halves (CE and ATUS: alternate years; DCPC:
   a random half of the people). Bin each comparison by its |z| in half A, then ask how often half B
   points the same way. A rule is useful if what it calls "clear" replicates almost always and what
   it calls "suggestive" replicates clearly better than a coin flip (50%).
2. Noise. Shuffle the weekday labels (within each household for CE, across people for DCPC and ATUS)
   so there is no weekday effect at all, and count how often each threshold still fires. With
   honest standard errors |z| >= 1 fires ~32% of the time and |z| >= 2 ~5%.

    uv run calibrate_significance.py          # with the SE scale factors in data_tools.SE_SCALE
    uv run calibrate_significance.py --raw    # without them (how the factors were fitted)
"""

import numpy as np
import pandas as pd

import data_tools as t

rng = np.random.default_rng(7)
# Run with --raw to see the unscaled numbers the scale factors were fitted from.
SCALE = {k: 1.0 for k in t.SE_SCALE} if "--raw" in __import__("sys").argv else t.SE_SCALE
DAYS = [2, 3, 4, 5, 6]
REPS = t.REPS


def z_scores_ce(day: pd.DataFrame, hh: pd.DataFrame, measure: str, split: str) -> list[float]:
    x = day[day.weekday.between(2, 6)].merge(hh, on=["NEWID", "year"])
    x["m"] = x.eval(t.CE_MEASURES[measure])
    out = []
    groups = [("all", x)] if split == "none" else x.groupby(split)
    for _, g in groups:
        if g.NEWID.nunique() < 300:
            continue
        cols = ["w"] + REPS
        num = g[cols].multiply(g.m, axis=0).groupby(g.weekday).sum()
        den = g[cols].groupby(g.weekday).sum()
        est = num / den  # rows: weekday, cols: main + replicates
        for d in DAYS:
            others = [k for k in DAYS if k != d]
            contrast = est.loc[d] - est.loc[others].mean()
            se = np.sqrt(((contrast[REPS] - contrast["w"]) ** 2).mean()) * SCALE["ce_diary"]
            out.append(contrast["w"] / se)
    return out


def z_scores_person(df: pd.DataFrame, ycol: str, split: str, cluster: str | None, scale: float = 1.0) -> list[float]:
    """Weighted weekday means with SEs clustered by person (DCPC) or one row per person (ATUS)."""
    out = []
    groups = [("all", df)] if split == "none" else df.groupby(split)
    for _, g in groups:
        if len(g) < 500:
            continue
        w, y = g.w.to_numpy(), g[ycol].to_numpy(float)
        cl = g[cluster].to_numpy() if cluster else np.arange(len(g))
        est, infl = {}, {}
        for d in DAYS:
            m = (g.weekday == d).to_numpy()
            W = w[m].sum()
            est[d] = (w[m] * y[m]).sum() / W
            e = np.zeros(len(g))
            e[m] = w[m] * (y[m] - est[d]) / W
            infl[d] = pd.Series(e).groupby(cl).sum()
        k = len(np.unique(cl))
        for d in DAYS:
            others = [o for o in DAYS if o != d]
            diff = est[d] - np.mean([est[o] for o in others])
            tot = infl[d] - sum(infl[o] for o in others) / 4
            se = np.sqrt((tot ** 2).sum() * k / (k - 1)) * scale
            out.append(diff / se)
    return out


def ce_all(day, hh):
    zs = []
    for measure in ["groceries", "eating_out", "fast_food", "full_service"]:
        for split in ["none", "region", "age_group", "earners", "income_group"]:
            zs += z_scores_ce(day, hh, measure, split)
    return np.array(zs)


def dcpc_all(df):
    zs = []
    for measure in t.DCPC_MEASURES:
        for split in ["none", "employment", "age_group"]:
            zs += z_scores_person(df, measure, split, "person")
    return np.array(zs)


def atus_all(df):
    zs = []
    for a in t.ATUS_ACTIVITIES:
        for split in ["none", "work_location", "age_group"]:
            zs += z_scores_person(df, f"{a}_any", split, None, SCALE["atus"])
    return np.array(zs)


def replication_table(name, za, zb):
    bins = [(0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, np.inf)]
    rows = []
    for lo, hi in bins:
        m = (np.abs(za) >= lo) & (np.abs(za) < hi)
        if m.sum() == 0:
            continue
        same = np.sign(za[m]) == np.sign(zb[m])
        strong = same & (np.abs(zb[m]) >= 1)
        rows.append({"source": name, "|z| in half A": f"{lo}-{hi}", "comparisons": int(m.sum()),
                     "same direction in half B": f"{100 * same.mean():.0f}%",
                     "same direction and |z|>=1 in half B": f"{100 * strong.mean():.0f}%"})
    return rows


def noise_table(name, z):
    return {"source": name, "comparisons": len(z),
            "|z|>=1": f"{100 * np.mean(np.abs(z) >= 1):.0f}%", "|z|>=1.5": f"{100 * np.mean(np.abs(z) >= 1.5):.0f}%",
            "|z|>=2": f"{100 * np.mean(np.abs(z) >= 2):.0f}%", "|z|>=3": f"{100 * np.mean(np.abs(z) >= 3):.0f}%"}


def main():
    ce_day, ce_hh = t.frame("ce_day"), t.frame("ce_hh")
    ce_hh = ce_hh[["NEWID", "year", "w", "region", "age_group", "earners", "income_group"] + REPS]
    dcpc = t.frame("dcpc_day")
    atus = t.frame("atus_day")
    atus = atus[atus.weekday.between(2, 6) & (atus.holiday != 1)]

    rep, noise = [], []

    # CE: alternate years are independent samples of households.
    a_years, b_years = [2015, 2017, 2019, 2022, 2024], [2016, 2018, 2021, 2023, 2025]
    za = ce_all(ce_day[ce_day.year.isin(a_years)], ce_hh)
    zb = ce_all(ce_day[ce_day.year.isin(b_years)], ce_hh)
    rep += replication_table("CE Diary", za, zb)
    shuf = ce_day[ce_day.year.isin(b_years) & ce_day.weekday.between(2, 6)].copy()
    shuf["weekday"] = shuf.groupby(["NEWID", "year"]).weekday.transform(lambda s: rng.permutation(s.to_numpy()))
    noise.append(noise_table("CE Diary (weekdays shuffled within household)", ce_all(shuf, ce_hh)))

    # DCPC: a random half of the people vs the other half.
    people = dcpc.person.unique()
    half = set(rng.choice(people, len(people) // 2, replace=False))
    ina = dcpc.person.isin(half)
    rep += replication_table("DCPC", dcpc_all(dcpc[ina]), dcpc_all(dcpc[~ina]))
    shuf = dcpc.copy()
    shuf["weekday"] = rng.permutation(shuf.weekday.to_numpy())
    noise.append(noise_table("DCPC (weekdays shuffled)", dcpc_all(shuf)))

    # ATUS: alternate years.
    za = atus_all(atus[atus.year % 2 == 1])
    zb = atus_all(atus[atus.year % 2 == 0])
    rep += replication_table("ATUS", za, zb)
    shuf = atus.copy()
    shuf["weekday"] = rng.permutation(shuf.weekday.to_numpy())
    noise.append(noise_table("ATUS (weekdays shuffled)", atus_all(shuf)))

    pd.set_option("display.width", 200)
    print("\nREPLICATION: does a weekday difference seen in one half show up in the other half?\n")
    print(pd.DataFrame(rep).to_string(index=False))
    print("\nNOISE: how often each threshold fires when there is no weekday effect at all\n")
    print(pd.DataFrame(noise).to_string(index=False))


if __name__ == "__main__":
    main()
