"""What each data source is, in words the agents read.

list_sources() gives the Lead the SUMMARY of every source so it can plan. describe_source() gives a
Researcher the DETAIL of one source: what it measures, which tool reads it, and what it can't do.
Hand-written from the scoping work in ../DATA_SCOPING.md.
"""

SOURCES = {
    "ce_diary": {
        "name": "BLS Consumer Expenditure Survey Diary (CE Diary), 2015-2025",
        "tool": "ce_diary_spending",
        "summary": "Household food DOLLARS by the weekday each purchase was made. About 11,000-13,000 "
                   "diary-weeks a year, nationally representative, with a New York metro subsample "
                   "(~700 diary-weeks a year). Groceries vs eating out (fast food, full service).",
        "detail": """\
What it is: households keep a 7-day diary of everything they buy; BLS records the weekday of each purchase.
Unit: one household (consumer unit) x one diary day. Every diary week covers each weekday exactly once,
so weekday comparisons are within the same households.
Measures (dollars per household per day): groceries (food at home), eating_out (fast food + full
service + vending/employer; excludes catered affairs), fast_food, full_service. Meal-type detail
(lunch_out, dinner_out, breakfast_out, snack_out) exists ONLY for 2015-2021.
Splits: period (2015-2019 vs 2022-2025), year, region, ny_metro, age_group (of the reference person),
earners (households with vs without anyone working), income_group (thirds).
Weights: FINLWT21; standard errors from the 44 BLS replicate weights (computed in code).
Can't do: individuals (it's households); who in the household bought it; work location or telework;
NYC proper (only the NY-Newark-Jersey City metro area); anything after 2025.
Caveats: 2020 is a pandemic year; 2022 changed how food away from home is coded (meal detail dropped).
About 1% of purchases have no recorded weekday and are excluded.""",
    },
    "dcpc": {
        "name": "Atlanta Fed Diary of Consumer Payment Choice (DCPC), 2021-2025",
        "tool": "dcpc_spending; dcpc_payday_window for the payday test",
        "summary": "Individual food DOLLARS by day from 3-day payment diaries (~4,000-5,500 people a year, "
                   "each October). Merchant types: grocery, sit-down restaurant, fast food. Has employment "
                   "and retirement status, so it can compare workers with retirees.",
        "detail": """\
What it is: each person records every payment for 3 consecutive days, with date, amount, merchant type.
Unit: one person x one diary day. Data are collected around October each year.
Measures (dollars per person per day, capped at each year's 99.5th percentile): groceries (grocery
stores, convenience stores without gas, pharmacies), full_service (sit-down restaurants and bars),
fast_food (fast food, coffee shops, cafeterias, food trucks), eating_out, food_total.
Splits: year, employment (full_time, part_time, retired, not_employed), age_group, ny_state (2022+).
Weights: daily_weight; standard errors clustered by person (computed in code).
Can't do: a full year (October only); pure food at grocery stores (includes pharmacy purchases);
NYC (only New York State, ~100 people a year); anyone's actual payday (the pay-date fields are
unreliable). dcpc_payday_window instead compares days that often follow a payday (1st-3rd, 15th-17th,
28th-31st) with other October days.
Caveats: smaller than the CE Diary, so year-by-year results are noisy; pooled 2021-2025 is better.""",
    },
    "nhanes": {
        "name": "CDC NHANES 'What We Eat in America' 24-hour recalls, 2017-2018 and 2021-2023",
        "tool": "nhanes_eating",
        "summary": "Where people's food came from (restaurant vs store) by weekday, from a nationally "
                   "representative food survey. No dollars. Shows the share of calories from restaurants "
                   "and the share of people who ate restaurant food, including at lunch vs dinner.",
        "detail": """\
What it is: people report everything they ate the previous day and where each food came from.
Unit: one person x one recall day. Precomputed design-weighted estimates with standard errors.
Measures (percent): pct_kcal_restaurant, pct_kcal_full_service, pct_kcal_fast_food, pct_kcal_store,
pct_kcal_workplace_cafeteria, pct_ate_restaurant, pct_ate_restaurant_lunch (10:30-14:00),
pct_ate_restaurant_dinner (17:00-21:00).
Groups: all, children_2_19, adults_20_64, employed_20_64, not_employed_20_64, age_65_plus.
Cycles: 2017-2018 (in-person recall) and 2021-2023 (phone recall); don't treat the difference
between cycles as a trend without saying the method changed.
Can't do: dollars; any geography (no state, city or region); dates or paydays; anything after 2023.
Caveats: Friday sample in 2021-2023 is small (~430 people), so subgroup Fridays are noisy.""",
    },
    "atus": {
        "name": "BLS American Time Use Survey (ATUS), 2015-2025",
        "tool": "atus_time_use; atus_payday_window for the payday test",
        "summary": "What people DID on each weekday: grocery shopping, buying prepared food, eating at "
                   "restaurants, cooking -- and WHERE they worked that day (office, home, mixed, no work). "
                   "The only source that links work location to food behavior. No dollars.",
        "detail": """\
What it is: one person reports every activity of one day, with where it happened.
Unit: one respondent x their single diary day (~6,000-9,000 people a year).
Measures: share of people who did it that day (grocery_any, buy_prepared_food_any, eat_restaurant_any,
eat_at_workplace_any, eat_home_any, food_prep_any) or average minutes (same names with _min).
work_location that day: office (60+ min of work at the workplace), home (60+ min of work at home),
mixed (both), elsewhere, no_work.
Splits: work_location, usual_teleworker (from the CPS interview months earlier; mostly 2023+),
age_group, period, year. Geography: us, ny_metro, nyc (NYC is very thin: ~30-50 employed weekday
diaries a year).
Diary dates cover every month, so atus_payday_window compares days that often follow a payday
(1st-3rd, 15th-17th, 28th-31st) with other days. Cooking (food_prep) and eating at home (eat_home)
show whether people cook less on the days they eat out more.
Can't do: dollars; anyone's actual payday; online vs in-store grocery shopping (coded the same); one
person's whole week.
Caveats: 2020 has no data mid-March to mid-May; October 2025 is missing (federal shutdown).
Holidays are excluded.""",
    },
    "mta": {
        "name": "MTA Subway Hourly Ridership (data.ny.gov), 2017-present",
        "tool": "mta_subway_ridership",
        "summary": "LIVE subway entries in NYC by weekday, systemwide and at 11 Manhattan office-district "
                   "stations. Shows the weekly office rhythm (hybrid work). No spending or food data.",
        "detail": """\
What it is: paid subway entries (OMNY taps and MetroCard swipes) by hour and station.
Station sets: office_11 (Grand Central, Times Sq, 47-50 Sts-Rockefeller Ctr, 5 Av/53 St,
Lexington Av/51-53 Sts, 34 St-Herald Sq, both 34 St-Penn Station complexes, Fulton St, both Wall St
stations) and systemwide.
Returns: each weekday as a percent of that year's Monday-Friday average, for a fall window
(Sept 15 - Nov 14, federal holidays excluded), optionally for the evening commute (4-7pm).
Can't do: food or spending; where people bought lunch; exits (entries only); fare evaders.
Caveats: compare weekday shapes within a year, not raw totals against 2019 (fare evasion has grown).""",
    },
}

REFERENCE_NOTES = {
    "office_attendance_nyc": """\
Cited office-attendance figures for New York City (not queryable data; quote with the link).
- Placer.ai: Friday's share of NYC weekday office visits fell from 17.2% (2019) to 11.5% (2023) and
  11.9% (2024); Tuesday was 24.6% in 2024. Source: Placer.ai, "Blueprint for Recovery" (Feb 27, 2025),
  https://www.placer.ai/
- Placer.ai national office index, Jan-Oct 2025: Tue 24.3%, Wed 23.7%, Thu 21.8%, Fri 12.4% of weekday
  office visits. Source: Placer.ai Office Index, October 2025, https://www.placer.ai/
- Kastle Systems Back to Work Barometer (badge swipes vs a Feb 2020 baseline): NYC weekly average
  72.2% in the week of Oct 27, 2026, with a Tuesday peak of 87.3%; in Sept 24-30, 2026 NYC's lowest
  day was Friday (34.9%) and highest Tuesday (81.1%). Source:
  https://www.kastle.com/safety-wellness/getting-america-back-to-work/ (quoted by hand; Kastle's
  terms forbid automated access).""",
    "hybrid_work_nyc": """\
- Partnership for New York City (March 2025 survey): on an average weekday 57% of Manhattan office
  workers are in the office (76% of pre-pandemic). In-office days per week: 10% five, 26% four,
  30% three, 16% two, 9% one, 8% fully remote. Source: https://pfnyc.org/research/""",
}


def summary_text() -> str:
    lines = []
    for key, s in SOURCES.items():
        lines.append(f"- {key}: {s['name']}. {s['summary']} Tool: {s['tool']}.")
    lines.append("- reference_note topics (cited quotes, not data): " + ", ".join(REFERENCE_NOTES))
    return "\n".join(lines)
