"""Food Week: deep research over US survey data on when, during the work week, people spend on food.

Built on the class deep-research pattern (the Wikipedia app this replaces): a Lead plans and decides,
Researchers run in parallel in fresh contexts, and a Writer turns their findings into a cited report.
What's ours:

  * The Lead runs an open agentic loop. It can ask the user to clarify, start rounds of parallel
    Researchers (at most 3 rounds: the first plus 2 follow-ups), and decides itself when there's
    enough to hand to the Writer -- which is one of its tools.
  * Researchers get only the data tools for the sources the Lead assigned them. Every tool reads one
    dataset, through fixed queries (see data_tools.py, mta.py). The model never writes SQL or SoQL.
  * Every query result gets an evidence ID. The Writer cites IDs; code checks every ID exists and
    every number in a cited sentence is in the evidence, and asks for one rewrite if not.
  * POST /chat keeps Project 1's shape: {response, session_id, tool_calls}. Every tool call by every
    agent is in tool_calls, with an `agent` field.
"""

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import uvicorn
from agents import Agent, ModelSettings, RunContextWrapper, Runner, function_tool, set_tracing_disabled
from agents.agent import StopAtTools
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import data_tools as dt
from catalog import SOURCES, summary_text
from context_panel import AgentContexts, PanelHooks, add_context_panel

load_dotenv(Path(__file__).parent / ".env")

# Traces upload to OpenAI by default, which needs an OpenAI key. We're on Gemini.
set_tracing_disabled(True)

# LiteLLM routes the call to Gemini. api_key is unused: Vertex AI uses your gcloud credentials.
MODEL_NAME = os.environ.get("MODEL", "vertex_ai/gemini-3.5-flash")
RESEARCHER_MODEL_NAME = os.environ.get("RESEARCHER_MODEL", "vertex_ai/gemini-3.5-flash-lite")
MODEL = LitellmModel(model=MODEL_NAME, api_key="unused")
RESEARCHER_MODEL = LitellmModel(model=RESEARCHER_MODEL_NAME, api_key="unused")
_vertex = {"vertex_location": os.environ.get("VERTEX_LOCATION", "global"), "timeout": 90, "num_retries": 2}
if os.environ.get("GOOGLE_CLOUD_PROJECT"):
    _vertex["vertex_project"] = os.environ["GOOGLE_CLOUD_PROJECT"]
SETTINGS = ModelSettings(extra_args=_vertex)
LEAD_SETTINGS = ModelSettings(extra_args=_vertex, parallel_tool_calls=True)

# --- Budget. Prompts ask for restraint; code enforces it. ---
MAX_ROUNDS = 3  # rounds of research: the first, then at most 2 follow-ups after the Lead reads findings
MAX_RESEARCHERS = 8  # in all, across rounds
MAX_PER_ROUND = 5
MAX_DATA_CALLS = 6  # per Researcher
MAX_MTA_CALLS = 6  # per run: the only tool that reaches the internet
# Cloud Run cuts a request at 600 s. The Lead gets 480 s; if it runs out, the Writer gets up to 90 s
# to report from whatever evidence exists, so the user still gets an answer inside the limit.
RUN_TIMEOUT = 480
WRITER_TIMEOUT = 90


@dataclass
class Research:
    """One run's state, shared by every agent and tool in it. The model never sees it."""

    hooks: PanelHooks
    question: str = ""
    lead_turns: int = 0  # how many model replies the Lead has made
    rounds: int = 0  # how many Lead turns started researchers
    round_turn: int = -1  # the Lead turn the current round belongs to
    in_round: int = 0  # researchers started in the current round
    researchers: int = 0
    data_calls: dict[str, int] = field(default_factory=dict)  # researcher -> data queries made
    mta_calls: int = 0
    findings: list[str] = field(default_factory=list)
    evidence: dict[str, dict] = field(default_factory=dict)  # "E3" -> {source, params, text, agent}
    tool_calls: list[dict] = field(default_factory=list)  # every call by every agent, for /chat
    outcome: dict | None = None  # {"type": "report"|"clarify", ...} once the Lead ends the run


class Hooks(PanelHooks):
    """The context panel's hooks, plus: count the Lead's turns (to group parallel research calls into
    rounds) and record every tool call with the agent that made it."""

    def __init__(self, contexts, emit):
        super().__init__(contexts, emit)
        self.run: Research | None = None

    async def on_llm_end(self, context, agent, response):
        await super().on_llm_end(context, agent, response)
        if agent.name == "Lead" and self.run:
            self.run.lead_turns += 1

    async def on_tool_end(self, context, agent, tool, result):
        if self.run is None:
            return
        try:
            args = json.loads(getattr(context, "tool_arguments", None) or "{}")
        except ValueError:
            args = {}
        text = str(result)
        self.run.tool_calls.append({"agent": agent.name, "name": tool.name, "args": args,
                                    "result": text if len(text) <= 6000 else text[:6000] + "\n[truncated]"})


def _agent_name(ctx) -> str:
    agent = getattr(ctx, "agent", None)
    return getattr(agent, "name", None) or "Researcher"


def _evidence(ctx: RunContextWrapper[Research], fn, **kwargs) -> str:
    """Run one data query, enforce the per-Researcher budget, and file the result as evidence."""
    run, who = ctx.context, _agent_name(ctx)
    used = run.data_calls.get(who, 0)
    if used >= MAX_DATA_CALLS:
        return f"Data-query limit reached ({MAX_DATA_CALLS}). Write your findings now from what you have."
    run.data_calls[who] = used + 1
    try:
        result: dt.Result = fn(**kwargs)
    except dt.ToolInputError as e:
        return json.dumps({"error": str(e), "hint": "Fix the arguments and try once more."})
    except Exception as e:  # the model can't see an exception: return something it can reason about
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:300]}"})
    eid = f"E{len(run.evidence) + 1}"
    run.evidence[eid] = {"source": result.source, "params": result.params, "text": result.text,
                         "caveats": result.caveats, "agent": who}
    return f"Evidence ID: {eid} (cite numbers from this result as [{eid}])\n{result.text}"


# --- Researcher tools: one per dataset, named after it ---


@function_tool
def describe_source(source: Literal["ce_diary", "dcpc", "nhanes", "atus", "mta"]) -> str:
    """Explain one dataset: what it measures, which tool reads it, its options, and what it can't do.

    Args:
        source: ce_diary, dcpc, nhanes, atus or mta.
    """
    s = SOURCES[source]
    return f"{s['name']}\nTool: {s['tool']}\n{s['detail']}"


@function_tool
def ce_diary_spending(
    ctx: RunContextWrapper[Research],
    measure: Literal["groceries", "eating_out", "fast_food", "full_service", "food_total",
                     "lunch_out", "dinner_out", "breakfast_out", "snack_out"],
    start_year: int,
    end_year: int,
    split_by: Literal["none", "period", "year", "region", "ny_metro", "age_group", "earners", "income_group"] = "none",
    geography: Literal["us", "ny_metro", "northeast"] = "us",
) -> str:
    """BLS CONSUMER EXPENDITURE DIARY ONLY. Household food dollars per day for each weekday (Mon-Fri),
    2015-2025, with standard errors and labeled weekday comparisons.

    Args:
        measure: groceries (food at home), eating_out, fast_food, full_service, food_total; or meal detail
            lunch_out / dinner_out / breakfast_out / snack_out (2015-2021 only).
        start_year: First year, 2015-2025.
        end_year: Last year, 2015-2025.
        split_by: Show a table per group: period (2015-2019, 2020-2021, 2022-2025), year, region, ny_metro
            (NY metro vs rest of US), age_group (of the household head), earners (households with/without
            a worker), income_group.
        geography: us (all), ny_metro (New York-Newark-Jersey City metro only), northeast.
    """
    return _evidence(ctx, dt.ce_spending, measure=measure, start_year=start_year, end_year=end_year,
                     split_by=split_by, geography=geography)


@function_tool
def dcpc_spending(
    ctx: RunContextWrapper[Research],
    measure: Literal["groceries", "eating_out", "fast_food", "full_service", "food_total"],
    start_year: int = 2021,
    end_year: int = 2025,
    split_by: Literal["none", "year", "employment", "age_group", "ny_state"] = "none",
) -> str:
    """ATLANTA FED DIARY OF CONSUMER PAYMENT CHOICE ONLY. Individual food dollars per person per day for
    each weekday (Mon-Fri), October diaries 2021-2025, with standard errors and labeled comparisons.

    Args:
        measure: groceries, eating_out, fast_food, full_service, food_total.
        start_year: First year, 2021-2025.
        end_year: Last year, 2021-2025.
        split_by: year, employment (full_time, part_time, retired, not_employed), age_group, or ny_state
            (New York State vs other states, 2022+).
    """
    return _evidence(ctx, dt.dcpc_spending, measure=measure, start_year=start_year, end_year=end_year, split_by=split_by)


@function_tool
def nhanes_eating(
    ctx: RunContextWrapper[Research],
    measure: Literal["pct_kcal_restaurant", "pct_kcal_full_service", "pct_kcal_fast_food", "pct_kcal_store",
                     "pct_kcal_workplace_cafeteria", "pct_ate_restaurant", "pct_ate_restaurant_lunch",
                     "pct_ate_restaurant_dinner"],
    cycle: Literal["2017-2018", "2021-2023", "both"] = "both",
    groups: list[Literal["all", "children_2_19", "adults_20_64", "employed_20_64", "not_employed_20_64",
                         "age_65_plus"]] | None = None,
) -> str:
    """CDC NHANES ONLY. Where people's food came from on each weekday (Mon-Fri): share of calories from
    restaurants or stores, or share of people who ate restaurant food (any time, lunch, dinner). No dollars.

    Args:
        measure: Which percentage to show.
        cycle: 2017-2018, 2021-2023 or both.
        groups: Up to 4 population groups (default: all).
    """
    return _evidence(ctx, dt.nhanes_eating, measure=measure, cycle=cycle, groups=list(groups or ["all"]))


@function_tool
def atus_time_use(
    ctx: RunContextWrapper[Research],
    activity: Literal["grocery", "buy_prepared_food", "eat_restaurant", "eat_at_workplace", "eat_home", "food_prep"],
    stat: Literal["share", "minutes"] = "share",
    start_year: int = 2022,
    end_year: int = 2025,
    split_by: Literal["none", "work_location", "usual_teleworker", "age_group", "period", "year"] = "none",
    geography: Literal["us", "ny_metro", "nyc"] = "us",
    employed_only: bool = False,
) -> str:
    """BLS AMERICAN TIME USE SURVEY ONLY. What people did on each weekday (Mon-Fri): grocery shopping,
    buying prepared food, eating at a restaurant/at work/at home, cooking -- optionally by where they
    worked that day. No dollars.

    Args:
        activity: Which food activity.
        stat: share (% who did it that day) or minutes (average minutes).
        start_year: First year, 2015-2025.
        end_year: Last year, 2015-2025.
        split_by: work_location (office, home, mixed, elsewhere, no_work that day), usual_teleworker,
            age_group, period, year.
        geography: us, ny_metro, or nyc (very small samples).
        employed_only: Only people with a job.
    """
    return _evidence(ctx, dt.atus_time_use, activity=activity, stat=stat, start_year=start_year, end_year=end_year,
                     split_by=split_by, geography=geography, employed_only=employed_only)


@function_tool
def mta_subway_ridership(
    ctx: RunContextWrapper[Research],
    year: int,
    station_set: Literal["office_11", "systemwide"] = "office_11",
    season: Literal["fall", "spring"] = "fall",
    hours: Literal["all_day", "morning_commute", "lunch", "evening_commute"] = "all_day",
) -> str:
    """MTA NYC SUBWAY RIDERSHIP ONLY (live from data.ny.gov). Subway entries for each weekday as an index
    (Mon-Fri average = 100): how the NYC office week is shaped. No food or spending data.

    Args:
        year: 2017 to this year.
        station_set: office_11 (11 Manhattan office-district stations) or systemwide.
        season: fall (Sep 15-Nov 14) or spring (Mar 1-Apr 30).
        hours: all_day, morning_commute (7-10am), lunch (11am-2pm), evening_commute (4-7pm).
    """
    run = ctx.context
    if run.mta_calls >= MAX_MTA_CALLS:
        return f"MTA query limit for this run reached ({MAX_MTA_CALLS}). Use the results you have."
    run.mta_calls += 1
    return _evidence(ctx, dt.nyc_ridership, year=year, station_set=station_set, season=season, hours=hours)


@function_tool
def reference_note(ctx: RunContextWrapper[Research],
                   topic: Literal["office_attendance_nyc", "hybrid_work_nyc"]) -> str:
    """Published figures on NYC office attendance and hybrid work (Placer.ai, Kastle, Partnership for NYC),
    quoted with their links. Not data you can query.

    Args:
        topic: office_attendance_nyc or hybrid_work_nyc.
    """
    return _evidence(ctx, dt.reference_note, topic=topic)


SOURCE_TOOLS = {"ce_diary": ce_diary_spending, "dcpc": dcpc_spending, "nhanes": nhanes_eating,
                "atus": atus_time_use, "mta": mta_subway_ridership, "reference": reference_note}

# --- The researcher: one narrow task, with only its assigned datasets' tools ---

researcher = Agent(
    name="Researcher",
    instructions="""\
You are a data researcher on a team studying WHEN during the work week (Monday-Friday) Americans spend
on food and eat out, and WHY. You get ONE task and the tools for the datasets assigned to you.

Each data tool reads exactly ONE dataset, and its name says which:
- ce_diary_spending -> BLS Consumer Expenditure Diary: household food DOLLARS by weekday, 2015-2025,
  nationally and for the New York metro area.
- dcpc_spending -> Atlanta Fed Diary of Consumer Payment Choice: individual food DOLLARS by weekday,
  October 2021-2025, by employment (full-time, retired...).
- nhanes_eating -> CDC NHANES: where people's food CAME FROM (restaurant vs store) by weekday. No dollars.
- atus_time_use -> BLS American Time Use Survey: food ACTIVITIES by weekday and by where people
  worked that day (office vs home). No dollars.
- mta_subway_ridership -> MTA subway entries: the NYC office week's shape. No food data.
- reference_note -> published NYC office-attendance figures to quote, with links.
You only have the tools for the datasets assigned to you. describe_source explains any dataset.

How to work:
1. Call describe_source once for each dataset you'll use.
2. Run at most 6 data queries. Choose options that answer your task directly, and use split_by to
   test explanations (e.g. atus_time_use split_by=work_location; dcpc_spending split_by=employment;
   ce_diary_spending split_by=period). Don't repeat a query.
3. Every result has Comparisons lines that label each weekday difference "clear", "suggestive" or
   "no clear difference". Use those words. Never call a "no clear difference" gap a difference.
4. Reply with at most 8 bullets. Each bullet: the finding in plain words, the key numbers copied
   exactly from the tool output with their unit, the label, and the evidence ID in brackets, e.g.
   "Eating out peaks on Friday: $11.12 vs $7.19 on Tuesday per household per day, a clear
   difference [E3]".
5. End with a bullet starting "Limits:" saying what your data can't show for this task.
If a tool returns an error, fix the call once; if it fails again, report what failed.""",
    model=RESEARCHER_MODEL,
    model_settings=SETTINGS,
    tools=[describe_source],
)


# --- The writer: called by the Lead, sees only the question, the brief, findings and evidence ---

writer = Agent(
    name="Writer",
    instructions="""\
You write the final report for a research tool about WHEN during the work week (Monday-Friday)
Americans spend on food and eat out, and WHY. Readers are curious and smart but not statisticians: a
cafe owner planning staff, a reporter, a student.

You get the question, the lead researcher's brief, the researchers' findings, and the evidence
records. Each record has an ID (E1, E2, ...), its source, its parameters and the exact table the
numbers came from. Use ONLY these. Never add facts from memory.

Write markdown with these sections:
1. "## The short answer": one paragraph that answers the question directly with the 2-3 numbers that
   matter most.
2. "## The overall picture": paint the whole picture. Don't walk through sources one by one. Explain
   what they show TOGETHER: when different surveys (dollars, calories, minutes; households, people)
   point the same way, say so, because agreement across independent surveys is the strongest evidence
   you have. When they disagree, say how and offer the likely reason (different units, years, people,
   or the October-only payment diary).
3. One "##" section per part of the question (for example "Groceries vs eating out", "Who differs",
   "New York City"). A small markdown table of weekday numbers helps when comparing days.
4. "## What explains it?": a table with one row per explanation the evidence touches (office days /
   hybrid work, an end-of-week habit, payday timing, a Monday restock, others the findings raise):
   | Explanation | Verdict | Evidence |, with Verdict one of Supported, Mixed, Not supported,
   Can't tell from this data, and one line of cited evidence.
5. "## What the data can't tell us": the specific limits that matter for this question.

Citations are mandatory:
- Every sentence (and table row) with a number or a factual finding ends with its evidence ID(s) in
  square brackets, e.g. [E3] or [E3, E7]. Use only IDs in the evidence records.
- Copy numbers exactly as they appear in the evidence. If you compare two numbers, show both
  ("$11.12 vs $7.19 [E3]") rather than computing new ones.
- The first time you use a survey, name it in plain words and give the unit: "the BLS Consumer
  Expenditure Diary, where households log every purchase" ... "per household per day".
- Say which years a number covers when the evidence table shows several (e.g. NHANES "2021-2023"
  vs "2017-2018" cycles, CE Diary periods): never present one cycle's number as if it were all years.

Describing differences: follow the evidence labels.
- clear -> "higher", "lower", "the highest day".
- suggestive -> "leans higher", "somewhat lower", "a hint that".
- no clear difference -> "about the same". Never call it a difference.
- Never claim cause: write "consistent with", "points toward", "doesn't fit". Never "causes",
  "drives", "because of", "proves", "due to".
- Keep places apart: national results are not New York results. The NY metro area (CE Diary), New
  York State (payment diary) and New York City (subway, time use) are different; name the one each
  number covers.
- Say who each finding speaks for. Write "Americans ..." or state a pattern flatly only when at least
  two surveys show it. When a finding rests on one survey, name that survey in the sentence ("in the
  payment diary, ...") and say no other source here confirms it. The Atlanta Fed payment diary is a
  national sample (not Atlanta residents), but it covers October only and is small, so a finding
  only it shows gets hedged wording ("leans", "a hint") even when its label is clear.

Style: plain, specific sentences; 500-900 words. Translate the tools' labels into plain words
("people 65 and older", not "age_65_plus"; "households with no one working", not "no_earners";
"the New York metro area", not "ny_metro"). No filler ("it is worth noting", "pivotal", "delve",
"in conclusion", "landscape").""",
    model=MODEL,
    model_settings=SETTINGS,
)


# --- Lead tools ---


@function_tool
def list_sources() -> str:
    """List every dataset the researchers can use: what it measures, years, places, and limits."""
    return summary_text()


def admit_researcher(run: Research) -> str | None:
    """Enforce the budget. All research calls in one Lead turn are one round. Returns a refusal or None."""
    if run.lead_turns != run.round_turn:  # the first research call of a new Lead turn opens a round
        if run.rounds >= MAX_ROUNDS:
            return f"No research rounds left (limit {MAX_ROUNDS}). Call write_report now."
        run.rounds += 1
        run.round_turn = run.lead_turns
        run.in_round = 0
    if run.researchers >= MAX_RESEARCHERS:
        return f"No researchers left (limit {MAX_RESEARCHERS}). Call write_report now."
    if run.in_round >= MAX_PER_ROUND:
        return f"At most {MAX_PER_ROUND} researchers per round. This task was not started."
    return None


@function_tool
async def research(
    ctx: RunContextWrapper[Research],
    task: str,
    sources: list[Literal["ce_diary", "dcpc", "nhanes", "atus", "mta", "reference"]],
) -> str:
    """Start one Researcher on one narrow task with the datasets you assign. Call this several times in
    the same turn to run researchers in parallel. Returns the Researcher's findings with evidence IDs.

    Args:
        task: One specific thing to find out, including the comparison to make, e.g. "Compare eating-out
            spending by weekday for 2015-2019 vs 2022-2025".
        sources: The 1-3 datasets this researcher should use.
    """
    run = ctx.context
    refusal = admit_researcher(run)
    if refusal:
        return refusal
    assigned = [s for s in dict.fromkeys(sources) if s in SOURCE_TOOLS][:3]
    if not assigned:
        return "Give at least one source: ce_diary, dcpc, nhanes, atus, mta or reference."
    run.researchers += 1
    run.in_round += 1
    number, this_round = run.researchers, run.rounds
    # What researcher.as_tool() does, written out: a fresh context per researcher, a numbered copy
    # with its own column in the context panel, and only the tools for its assigned datasets.
    agent = researcher.clone(name=f"Researcher {number}", tools=[describe_source] + [SOURCE_TOOLS[s] for s in assigned])
    prompt = f"Task: {task}\nDatasets assigned: {', '.join(assigned)}\nOverall question (for context): {run.question}"
    try:
        result = await Runner.run(agent, prompt, context=run, hooks=run.hooks, max_turns=10)
        findings = str(result.final_output)
    except Exception as e:
        findings = f"Researcher failed: {type(e).__name__}: {str(e)[:200]}"
    run.findings.append(f"### Researcher {number} (round {this_round}): {task}\nDatasets: {', '.join(assigned)}\n\n{findings}")
    left = f"[Round {this_round} of at most {MAX_ROUNDS}; {MAX_RESEARCHERS - run.researchers} researchers left]"
    return f"{findings}\n\n{left}"


@function_tool
def ask_user(ctx: RunContextWrapper[Research], question: str, suggestions: list[str]) -> str:
    """Ask the user to clarify a vague question. This ends the run; their answer starts a new one.

    Args:
        question: One short clarifying question.
        suggestions: 2-4 sharper versions of their question they could pick.
    """
    ctx.context.outcome = {"type": "clarify", "question": question, "suggestions": suggestions[:4]}
    return "Asked the user."


@function_tool
async def write_report(ctx: RunContextWrapper[Research], brief: str) -> str:
    """Hand the research to the Writer, who writes the final cited report. Call this once, when the
    findings answer the question (or no research rounds are left). This ends the run.

    Args:
        brief: What the report should lead with, where the sources agree or conflict, and which parts
            of the question the data can't answer.
    """
    run = ctx.context
    if not run.findings:
        return "There are no findings yet: call research first."
    report = await write_and_check(run, brief)
    run.outcome = {"type": "report", "markdown": report}
    return "Report written."


lead = Agent(
    name="Lead",
    instructions=f"""\
You lead a research team that answers questions about WHEN during the work week (Monday-Friday)
Americans spend money on food -- groceries and eating out -- and WHY, using US survey data, with New
York City as a case study. You plan, delegate and judge. You never query data yourself and never write
the report yourself.

Tools: list_sources, research (starts one Researcher; call it several times in one turn to run them in
parallel), ask_user, write_report (hands everything to the Writer and ends the run).

1. Decide whether the question is answerable here.
   - Off-topic (nothing to do with food spending, eating out, grocery shopping or the work-week rhythm
     behind them): reply in two sentences that you only cover weekday food spending, with two example
     questions. Call no tools.
   - Vague: call ask_user when you can't tell what to measure, compare or for where, e.g. "what do people
     eat?", "tell me about NYC", "is Friday different?". Ask one short question and give 2-4 sharper
     rewrites. Don't ask when a sensible reading exists: "Which weekday do Americans spend the most on
     food?" is clear (cover groceries and eating out, nationally). If the message already contains the
     user's clarification, don't ask again.
   - Weekends: this tool covers Monday-Friday. Say so in your brief and answer the weekday part.
2. Call list_sources once, then plan.
3. Start a round of 2-5 Researchers in ONE turn. Give each one narrow task (one measure, one comparison)
   and the 1-3 datasets that fit it. Spread the work across kinds of evidence: dollars (ce_diary, dcpc),
   eating behavior (nhanes), activities and work location (atus), the NYC office week (mta, reference),
   New York spending (ce_diary with ny_metro). Where the question asks why, set tasks that test the
   candidate explanations: office days vs home days (hybrid work), an end-of-week habit that shows up
   even for people who don't work, payday timing, a Monday restock.
4. Read the findings and decide. If part of the question has no evidence, sources disagree, or a finding
   begs an obvious follow-up (e.g. "does the Friday spike hold for retirees?"), start another round
   aimed at just that gap. You have at most {MAX_ROUNDS} rounds in all (the first plus {MAX_ROUNDS - 1}
   follow-ups) and {MAX_RESEARCHERS} researchers; research tells you what's left. Stop early when the
   evidence already answers the question -- more rounds are not better.
5. Call write_report with a brief: the main answer, the strongest evidence (especially where several
   surveys agree), conflicts to explain, and what the data can't answer. Then stop.""",
    model=MODEL,
    model_settings=LEAD_SETTINGS,
    tools=[list_sources, research, ask_user, write_report],
    tool_use_behavior=StopAtTools(stop_at_tool_names=["ask_user", "write_report"]),
)


# --- Checking the Writer's citations ---

CITE = re.compile(r"\[(E\d+(?:\s*,\s*E\d+)*)\]")
NUMBER = re.compile(r"(?<![\w.])\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?%?")


def _numbers_in(text: str) -> set[float]:
    out = set()
    for m in NUMBER.finditer(text):
        try:
            out.add(float(m.group(1).replace(",", "") + (m.group(2) or "")))
        except ValueError:
            pass
    return out


def check_citations(report: str, evidence: dict[str, dict]) -> list[str]:
    """Problems a reader would care about: unknown evidence IDs, and numbers in a cited sentence that
    don't appear in the evidence it cites (allowing rounding, and index-to-percent: 133 -> '33% more')."""
    problems = []
    for eid in sorted({e.strip() for m in CITE.finditer(report) for e in m.group(1).split(",")}):
        if eid not in evidence:
            problems.append(f"{eid} is cited but no such evidence exists.")
    for sentence in re.split(r"(?<=[.!?])\s+|\n", report):
        ids = [e.strip() for m in CITE.finditer(sentence) for e in m.group(1).split(",")]
        if not ids:
            continue
        pool = set()
        for eid in ids:
            if eid in evidence:
                pool |= _numbers_in(evidence[eid]["text"])
                pool |= {abs(v - 100) for v in _numbers_in(evidence[eid]["text"]) if 50 <= v <= 250}
        claim = CITE.sub("", sentence)
        for v in _numbers_in(claim):
            if v in (1, 2, 3, 4, 5, 6, 7) or 1990 <= v <= 2030:  # counts like "two surveys" and years
                continue
            if not any(abs(v - p) <= max(0.051, 0.006 * abs(p)) or round(p) == v for p in pool):
                problems.append(f"'{v:g}' in \"{claim.strip()[:120]}\" is not in {', '.join(ids)}.")
    return problems


CAUSAL = re.compile(r"\b(causes?|caused by|causing|drives|driven by|driving|because of|due to|proves?|"
                    r"explains why|(?:does|do) not explain|doesn't explain|don't explain|leads? to)\b", re.I)


MAX_WORDS = 1100  # the prompt asks for 500-900; tables add words, so only flag clear overruns
RAW_LABEL = re.compile(r"\b(age_65_plus|65_plus|under_35|35_54|55_64|no_earners|has_earners|ny_metro|ny_state|"
                       r"employed_20_64|not_employed_20_64|adults_20_64|children_2_19|rest_of_us|full_time|part_time|"
                       r"not_employed|no_work|work_location|pct_\w+)\b")


def check_wording(report: str) -> list[str]:
    """Causal phrasing the data can't support (the section title "What explains it?" is fine), raw tool
    labels, and length."""
    problems = []
    body = report.split("## Sources")[0]
    words = len(body.split())
    if words > MAX_WORDS:
        problems.append(f"The report is {words} words; cut it to at most 900 by trimming repetition, keeping every section.")
    labels = sorted(set(RAW_LABEL.findall(body)))
    if labels:
        problems.append(f"Replace raw data labels with plain words: {', '.join(labels)}.")
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        for m in CAUSAL.finditer(line):
            problems.append(f'Causal wording "{m.group(0)}" in "{line.strip()[:100]}": use "consistent with", '
                            f'"fits", "doesn\'t fit" or "points toward" instead.')
    return problems


def evidence_for_writer(run: Research) -> str:
    blocks = []
    for eid, e in run.evidence.items():
        name = dt.SOURCE_LINKS[e["source"]][0]
        blocks.append(f"[{eid}] {name} | {json.dumps(e['params'])}\n{e['text']}")
    return "\n\n".join(blocks)


async def write_and_check(run: Research, brief: str) -> str:
    prompt = (f"Question: {run.question}\n\nLead's brief:\n{brief}\n\n# Findings\n\n" + "\n\n".join(run.findings)
              + "\n\n# Evidence records\n\n" + evidence_for_writer(run))
    result = await Runner.run(writer, prompt, context=run, hooks=run.hooks, max_turns=2)
    report = str(result.final_output)
    problems = check_citations(report, run.evidence)
    wording = check_wording(report)
    if problems or wording:  # one rewrite, with the exact problems
        fix = ("Your report has problems. Fix each one. For a number: cite the evidence that contains it, copy it "
               "exactly, or remove it. For wording: rephrase without claiming cause. Return the full corrected "
               "report.\n- " + "\n- ".join((problems + wording)[:20]))
        result = await Runner.run(writer, result.to_input_list() + [{"role": "user", "content": fix}],
                                  context=run, hooks=run.hooks, max_turns=2)
        report = str(result.final_output)
        problems = check_citations(report, run.evidence)
    return report + sources_section(report, run) + (
        "\n\n> **Citation check:** " + str(len(problems)) + " number(s) couldn't be matched to the cited evidence: "
        + "; ".join(problems[:5]) if problems else "")


def sources_section(report: str, run: Research) -> str:
    cited = []
    for m in CITE.finditer(report):
        for eid in m.group(1).split(","):
            eid = eid.strip()
            if eid in run.evidence and eid not in cited:
                cited.append(eid)
    if not cited:
        return ""
    lines = ["\n\n## Sources"]
    for eid in sorted(cited, key=lambda e: int(e[1:])):
        e = run.evidence[eid]
        name, url = dt.SOURCE_LINKS[e["source"]]
        params = ", ".join(f"{k}={v}" for k, v in e["params"].items())
        link = f" ({url})" if url else ""
        lines.append(f"- **[{eid}]** {name}{link}: {params}")
    return "\n".join(lines)


# --- One research run ---


async def deep_research(question: str, hooks: Hooks) -> Research:
    run = Research(hooks=hooks, question=question)
    hooks.run = run
    final, timed_out = "", False
    try:
        result = await asyncio.wait_for(Runner.run(lead, question, context=run, hooks=hooks, max_turns=14), RUN_TIMEOUT)
        final = str(result.final_output)
    except asyncio.TimeoutError:
        timed_out = True
    if run.outcome is None:
        if run.findings:  # the Lead stopped (or ran out of time) without handing off: write from what there is
            note = ("Research ran out of time. Answer from the findings so far and say the research was cut short."
                    if timed_out else "The lead stopped without a brief. Answer from the findings.")
            report = await asyncio.wait_for(write_and_check(run, note), WRITER_TIMEOUT)
            run.outcome = {"type": "report", "markdown": report}
        elif timed_out:
            run.outcome = {"type": "answer", "markdown": "The research ran out of time before finding anything. Try a narrower question."}
        else:  # off-topic, or a direct answer
            run.outcome = {"type": "answer", "markdown": final}
    return run


def outcome(run: Research) -> dict:
    return run.outcome or {"type": "answer", "markdown": "The research ended without a result. Try rephrasing."}


def response_text(run: Research) -> str:
    o = outcome(run)
    if o["type"] == "clarify":
        options = "\n".join(f"- {s}" for s in o["suggestions"])
        return f"{o['question']}\n\nFor example:\n{options}"
    return o["markdown"]


def compose_question(message: str, pending: str | None) -> str:
    """A reply to a clarifying question carries the original question with it."""
    if pending:
        return f"{pending}\n\nClarification from the user: {message}"
    return message


# --- FastAPI ---

HERE = Path(__file__).parent
app = FastAPI(title="Food Week")
panels: dict[str, AgentContexts] = {}  # run_id -> every agent's context, for the context panel
add_context_panel(app, panels)
sessions: dict[str, dict] = {}  # session_id -> {"pending": original question awaiting clarification}
MAX_SESSIONS = 500


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


def _session(session_id: str | None) -> tuple[str, dict]:
    sid = session_id or str(uuid.uuid4())
    if sid not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.pop(next(iter(sessions)))
        sessions[sid] = {"pending": None}
    return sid, sessions[sid]


@app.get("/")
def index():
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "researcher_model": RESEARCHER_MODEL_NAME,
            "data": dt.DB_PATH.exists()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Project 1's shape. Runs the whole research and returns the report (or a clarifying question)."""
    sid, session = _session(request.session_id)
    question = compose_question(request.message.strip()[:2000], session["pending"])
    run_id = str(uuid.uuid4())
    panels[run_id] = AgentContexts()
    hooks = Hooks(panels[run_id], lambda _event: None)  # /chat doesn't stream
    try:
        run = await deep_research(question, hooks)
        session["pending"] = question if outcome(run)["type"] == "clarify" else None
        return ChatResponse(response=response_text(run), session_id=sid, tool_calls=run.tool_calls)
    except Exception as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else ""
        calls = hooks.run.tool_calls if hooks.run else []
        return ChatResponse(response=f"Research failed: {type(e).__name__}: {first[:300]}", session_id=sid, tool_calls=calls)
    finally:
        panels.pop(run_id, None)


@app.post("/clear")
def clear(session_id: str | None = None):
    if session_id:
        sessions.pop(session_id, None)
    return {"status": "ok"}


@app.get("/run")
async def run_stream(question: str, session_id: str | None = None):
    """The same research as /chat, streamed as Server-Sent Events for the web page."""
    sid, session = _session(session_id)
    full_question = compose_question(question.strip()[:2000], session["pending"])
    run_id = str(uuid.uuid4())
    panels[run_id] = AgentContexts()
    if len(panels) > 50:
        panels.pop(next(iter(panels)))
    queue: asyncio.Queue = asyncio.Queue()
    hooks = Hooks(panels[run_id], queue.put_nowait)

    async def work():
        try:
            run = await deep_research(full_question, hooks)
            result = outcome(run)
            session["pending"] = full_question if result["type"] == "clarify" else None
            queue.put_nowait({"type": "done", "kind": result["type"], "markdown": response_text(run),
                              "suggestions": result.get("suggestions", []), "session_id": sid,
                              "tool_calls": run.tool_calls, "rounds": run.rounds, "researchers": run.researchers})
        except Exception as e:
            queue.put_nowait({"type": "error", "message": f"{type(e).__name__}: {str(e)[:500]}"})

    task = asyncio.create_task(work())

    async def events():
        try:
            yield f"data: {json.dumps({'type': 'start', 'run_id': run_id, 'session_id': sid})}\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event["type"] in ("done", "error"):
                    break
        finally:
            task.cancel()  # the page closed the stream: stop spending tokens

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
