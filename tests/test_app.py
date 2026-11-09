"""Tests that never call Gemini: data tools run for real on data/food_week.duckdb, the model is mocked.

Run:  uv run --group dev pytest -q
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import data_tools as dt  # noqa: E402

# --- data tools ---


def test_ce_friday_eating_out_is_clearly_highest():
    r = dt.ce_spending("eating_out", 2022, 2025, "none", "us")
    assert "Fri vs other weekdays" in r.text and "-> clear" in r.text.split("Fri vs other weekdays")[1].splitlines()[0]


def test_meal_detail_rejected_after_2021():
    with pytest.raises(dt.ToolInputError):
        dt.ce_spending("lunch_out", 2022, 2025, "none", "us")


def test_out_of_range_years_rejected():
    with pytest.raises(dt.ToolInputError):
        dt.dcpc_spending("food_total", 2010, 2012, "none")


def test_thin_cells_are_suppressed():
    r = dt.atus_time_use("eat_restaurant", "share", 2025, 2025, "work_location", "nyc", True)
    assert "too thin to report" in r.text


def test_payday_window_compares_within_weekday():
    r = dt.payday_window("atus", "eat_restaurant", "share", 2015, 2025)
    assert "1st-3rd" in r.text and "15th-17th" in r.text and "other days" in r.text
    assert "Nobody's actual payday is recorded" in r.text
    with pytest.raises(dt.ToolInputError):
        dt.payday_window("ce_diary", "eating_out", "share", 2015, 2025)


def test_researchers_get_every_tool_for_their_dataset():
    names = {t.name for t in app_module.SOURCE_TOOLS["atus"]}
    assert names == {"atus_time_use", "atus_payday_window"}


def test_mta_rejects_bad_year_without_network():
    with pytest.raises(dt.ToolInputError):
        dt.nyc_ridership(1999, "office_11", "fall", "all_day")


def test_mta_query_is_built_from_constants():
    import mta

    captured = {}

    def fake_get(url, params, headers, timeout, follow_redirects):
        captured.update(url=url, params=params)
        return SimpleNamespace(status_code=200, content=b"[]", json=lambda: [])

    with patch.object(mta.httpx, "get", side_effect=fake_get):
        with pytest.raises(mta.RidershipError):  # no rows -> too few days
            mta._cache.clear()
            mta.weekday_shares(2025, "office_11", "fall", "lunch")
    assert captured["url"] == "https://data.ny.gov/resource/5wq4-mkjj.json"
    assert "'610'" in captured["params"]["$where"] and "between 11 and 13" in captured["params"]["$where"]


# --- citation checker ---

EVIDENCE = {"E1": {"source": "ce_diary", "params": {}, "text": "Fri | 11.12 | 0.16 | 46,220 | 133\nTue | 7.19 | 0.13 | 46,220 | 86"}}


def test_citations_ok():
    report = "Eating out peaks Friday at $11.12 vs $7.19 on Tuesday [E1]. That is 33% above average [E1]."
    assert app_module.check_citations(report, EVIDENCE) == []


def test_citations_catch_unknown_id_and_made_up_number():
    report = "Friday is $14.50 [E1]. Monday is $3 [E9]."
    problems = app_module.check_citations(report, EVIDENCE)
    assert any("E9" in p for p in problems) and any("14.5" in p for p in problems)


def test_peak_and_low_claims_must_match_the_table():
    ev = {"E1": {"text": dt.nhanes_eating("pct_kcal_restaurant", "2021-2023", ["all"]).text},
          "E3": {"text": dt.ce_spending("eating_out", 2022, 2025, "none", "us").text}}
    wrong = "Restaurant calories are low on Tuesday at 17.24% [E1]."  # Wednesday (16.97) is lower
    right = "Spending is lowest on Tuesday at $7.19 [E3] and peaks on Friday at $11.12 [E3]."
    assert app_module.check_extremes(wrong, ev)
    assert app_module.check_extremes(right, ev) == []


def test_causal_wording_is_flagged():
    assert app_module.check_wording("Hybrid work does not explain the spike [E1].")
    assert app_module.check_wording("Friday dining is driven by paydays.")
    assert not app_module.check_wording("## What explains it?\nThe spike is consistent with an end-of-week habit [E1].")


def test_raw_labels_and_length_are_flagged():
    assert any("age_65_plus" in p for p in app_module.check_wording("People in age_65_plus eat out [E1]."))
    assert any("words" in p for p in app_module.check_wording("word " * 1200))


# --- the research tool's budget ---


def _admit_round(run, k):
    """k parallel research calls in one new Lead turn; returns how many were admitted."""
    run.lead_turns += 1
    admitted = 0
    for _ in range(k):
        if app_module.admit_researcher(run) is None:
            run.researchers += 1
            run.in_round += 1
            admitted += 1
    return admitted


def test_parallel_calls_in_one_turn_are_one_round():
    run = app_module.Research(hooks=None, question="q")
    assert _admit_round(run, 3) == 3 and run.rounds == 1


def test_at_most_three_rounds():
    run = app_module.Research(hooks=None, question="q")
    assert [_admit_round(run, 2) for _ in range(4)] == [2, 2, 2, 0]
    assert run.rounds == 3


def test_researcher_and_per_round_caps():
    run = app_module.Research(hooks=None, question="q")
    assert _admit_round(run, 7) == app_module.MAX_PER_ROUND
    assert _admit_round(run, 5) == app_module.MAX_RESEARCHERS - app_module.MAX_PER_ROUND


# --- /chat keeps Project 1's shape ---


def test_chat_shape_and_agent_field():
    async def fake_research(question, hooks, history=None):
        run = app_module.Research(hooks=hooks, question=question)
        hooks.run = run
        run.tool_calls = [{"agent": "Researcher 1", "name": "ce_diary_spending", "args": {"measure": "eating_out"}, "result": "..."}]
        run.outcome = {"type": "report", "markdown": "Friday [E1]."}
        return run

    with patch.object(app_module, "deep_research", side_effect=fake_research):
        r = TestClient(app_module.app).post("/chat", json={"message": "Which weekday do people eat out most?"})
    body = r.json()
    assert set(body) == {"response", "session_id", "tool_calls"}
    assert body["tool_calls"][0]["agent"] == "Researcher 1"


def test_clarification_carries_the_original_question():
    seen = []

    async def fake_research(question, hooks, history=None):
        seen.append(app_module.lead_input(question, history or []))
        run = app_module.Research(hooks=hooks, question=question)
        hooks.run = run
        run.outcome = ({"type": "clarify", "question": "Groceries or eating out?", "suggestions": ["a", "b"]}
                       if len(seen) == 1 else {"type": "report", "markdown": "ok"})
        return run

    with patch.object(app_module, "deep_research", side_effect=fake_research):
        client = TestClient(app_module.app)
        first = client.post("/chat", json={"message": "food on fridays?"}).json()
        client.post("/chat", json={"message": "eating out", "session_id": first["session_id"]})
    assert "Groceries or eating out?" in first["response"]
    assert "User: food on fridays?" in seen[1] and "You asked:\nGroceries or eating out?" in seen[1]
    assert seen[1].endswith("# New question\n\neating out")


def test_index_explains_the_tool():
    r = TestClient(app_module.app).get("/")
    assert "What it explores" in r.text and "Who it's for" in r.text and "How to use it" in r.text


def test_explains_table_keeps_only_tested_rows():
    report = (
        "## What explains it?\n"
        "| Explanation | Verdict | Evidence |\n|---|---|---|\n"
        "| Payday timing | Mixed | $12.32 vs $10.41 [E2] |\n"
        "| Verified paydays | Can't tell from this data | No payroll records [E1] |\n"
        "| Monday restock | **Not tested** | CE groceries by weekday would test it |\n"
        "| Work location | Can’t tell | None link it [E27] |\n"
        "\n## What the data can't tell us\n"
        "| Thing | Can't tell | kept outside the table |\n"
    )
    out = app_module.drop_untested_rows(report)
    assert "Payday timing" in out
    assert "Verified paydays" not in out and "Monday restock" not in out and "Work location" not in out
    assert "kept outside the table" in out


def test_question_box_clears_on_submit():
    html = (Path(app_module.__file__).parent / "index.html").read_text()
    assert '$("question").value = "";' in html



def test_follow_ups_carry_the_conversation():
    seen = []

    async def fake_research(question, hooks, history=None):
        seen.append(app_module.lead_input(question, history or []))
        run = app_module.Research(hooks=hooks, question=question)
        hooks.run = run
        run.outcome = {"type": "report", "markdown": "Friday is highest at $9.73 [E3].\n\n## Sources\n- [E3] CE"}
        return run

    with patch.object(app_module, "deep_research", side_effect=fake_research):
        client = TestClient(app_module.app)
        sid = client.post("/chat", json={"message": "which weekday for eating out?"}).json()["session_id"]
        assert client.get("/history", params={"session_id": sid}).json()["turns"][0]["question"] == "which weekday for eating out?"
        client.post("/chat", json={"message": "what about lunch?", "session_id": sid})
    assert seen[0] == "which weekday for eating out?"  # nothing earlier: just the question
    assert "which weekday for eating out?" in seen[1] and "Friday is highest at $9.73." in seen[1]
    assert "[E3]" not in seen[1] and "## Sources" not in seen[1]
    assert seen[1].endswith("# New question\n\nwhat about lunch?")
    assert client.get("/history", params={"session_id": "nope"}).json()["turns"] == []


def test_label_wording_is_flagged():
    assert any("clear difference" in p for p in app_module.check_wording("## X\nDinner climbs to 28.35%, a clear difference [E22].\n"))
    assert not app_module.check_wording("## X\nThe days show no clear difference [E1].\n")
