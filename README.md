# Food Week: deep research on when Americans spend on food

**When during the work week (Monday–Friday) do Americans spend money on food (groceries and eating out), and why?** Ask a question and a team of agents answers it from five US datasets, with every number cited to the query that produced it. New York City is the case study.

**Who it's for:** café and restaurant owners planning staff and promotions by weekday, reporters covering hybrid work, and anyone curious about everyday spending habits.

**How to use it:** open the page, type a question about Monday–Friday food spending or eating (or click an example), and press Research. A run takes 1–3 minutes. The activity list and the context panel on the right show each agent working. Vague questions get a short follow-up question with suggested rewrites. Follow-ups build on the conversation: the question box shows the earlier turns the model will see with your next question (or says it has no earlier context), and **Start over** clears them.

Example questions:
- Which weekday do Americans spend the most on eating out, and why?
- Do people who work from home buy groceries on different days than office workers?
- Is the Friday food-spending spike just a payday effect, or do retirees show it too?
- Does the New York area follow the same weekday food pattern as the rest of the country?

## How it works

```
question ─▶ LEAD ──(off-topic)──▶ short refusal
              │ ──(vague)──────▶ ask_user: a question + 2-4 sharper rewrites; the reply resumes the run
              │
              │  research() × 2-5 in ONE turn  ─▶  Researcher 1 … N run in parallel, fresh contexts,
              │                                    each with only its assigned datasets' tools
              │◀── findings with evidence IDs [E1], [E2] …
              │
              │  the Lead decides: gap or conflict? ─▶ another round (at most 3 rounds in all)
              │                    enough?          ─▶ write_report(brief)
              ▼
            WRITER (question + brief + findings + evidence) ─▶ code checks citations ─▶ report + Sources
```

The pattern comes from the class Wikipedia deep-research app (same Agents SDK + LiteLLM/Vertex setup, same context panel). What changed:

- **An open agentic loop.** The Lead isn't on a script. Each turn it decides what to do next: clarify, start a round of parallel Researchers, start a follow-up round for a gap, or hand off to the Writer. **The Writer is one of the Lead's tools** (`write_report`), so the Lead decides when there's enough. If it stops without handing off, code calls the Writer anyway.
- **Rounds, not researchers, are capped at 3**: the first round, then the Lead, then a follow-up round, then the Lead, then one more. All `research` calls in one Lead turn count as one round. Code enforces this in `admit_researcher()`, not just the prompt.
- **One Researcher type, many copies.** Each `research` call clones it with a number (its own column in the context panel) and gives it **only the tools for the datasets the Lead assigned**.

### Agents and their tools

| Agent | Model | Tools |
|---|---|---|
| **Lead** (manager) | `gemini-3.5-flash` | `list_sources`: what each dataset can and can't do · `research(task, sources)`: start one Researcher (call several in one turn for parallel) · `ask_user(question, suggestions)`: clarify a vague question; ends the run · `write_report(brief)`: run the Writer; ends the run |
| **Researcher** (one type, run in parallel) | `gemini-3.5-flash-lite` | `describe_source` plus, for its assigned datasets only: `ce_diary_spending` (BLS CE Diary) · `dcpc_spending` and `dcpc_payday_window` (Atlanta Fed DCPC) · `nhanes_eating` (CDC NHANES) · `atus_time_use` and `atus_payday_window` (BLS ATUS) · `mta_subway_ridership` (MTA, live) · `reference_note` (cited office-attendance figures) |
| **Writer** | `gemini-3.5-flash` | none: sees only the question, the Lead's brief, the findings and the evidence records |

Each data tool reads **one** dataset and is named after it. The two payday tools compare days that often follow a payday (1st–3rd, 15th–17th, 28th–31st) with other days of the month on the same weekday; no survey records anyone's actual payday. The docstring starts with the dataset in capitals (e.g. "BLS AMERICAN TIME USE SURVEY ONLY"), and the Researcher prompt maps each tool to its dataset.

## The data

| Source | What it measures | Years | Where |
|---|---|---|---|
| **BLS Consumer Expenditure Survey Diary** | Household food **dollars** by the weekday of purchase: groceries, fast food, full service; lunch/dinner detail through 2021 | 2015–2025 | US, regions, NY metro |
| **Atlanta Fed Diary of Consumer Payment Choice** | Individual food **dollars** by day: grocery, sit-down, fast food; by employment and retirement | Oct 2021–2025 | US, NY State |
| **CDC NHANES (What We Eat in America)** | Where food came from (restaurant vs store) by weekday; lunch vs dinner; **no dollars** | 2017–18, 2021–23 | US only |
| **BLS American Time Use Survey** | Grocery shopping, buying prepared food, eating out, cooking, by **where people worked that day**; **no dollars** | 2015–2025 | US, NY metro, NYC (thin) |
| **MTA Subway Hourly Ridership** (live API) | Subway entries by weekday at 11 Manhattan office stations or systemwide | 2017–now | NYC |

Cited office-attendance figures (Placer.ai, Kastle, Partnership for NYC) are hand-copied notes with links, not queried data. Kastle's terms forbid automated access.

### Offline build

The raw files (~2 GB of SAS, XPT, CSV and fixed-width data in `../local data/`) are turned into one 40 MB DuckDB file by `build_data.py`. All the error-prone survey work happens there, once, in code:
- linking each purchase to its weekday, and food spending codes to categories via the BLS stub files
- weights and replicate weights
- NHANES design-based standard errors
- ATUS work location per day

```bash
uv run --group build build_data.py          # reads ../local data, writes data/food_week.duckdb
uv run --group build build_data.py --fresh  # ignore the cached intermediate tables
```

Checks run during the build:
- **CE grocery dollars:** match BLS's own household totals within 1.4%. The gap is purchases with no recorded weekday.
- **CE eating out:** deliberately excludes catered affairs (weddings, parties), which are about 15% of BLS food away from home.
- **CE weekday coverage:** every diary week covers each weekday exactly once (879,998 day rows = 125,714 diary weeks × 7), so CE weekday comparisons are within the same households.

`data/food_week.duckdb` is committed and copied into the Cloud Run image; the raw files are not. CE, ATUS and NHANES are US-government public-domain data; DCPC is a free public release from the Atlanta Fed.

## Guardrails

**Input**
- **Off-topic** questions get a two-sentence refusal with example questions.
- **Vague** questions ("what about NYC?") make the Lead call `ask_user` with a question and 2–4 sharper rewrites. The session remembers the original question, so the reply continues it. The Lead doesn't ask when a sensible reading exists, and never asks twice.
- **Weekends** are out of scope; the Lead says so and answers the weekday part.
- Questions are cut at 2,000 characters.

**Tools: no model-written SQL anywhere**
- Every tool argument is an enum or a bounded number (JSON-schema enums from `Literal` types, re-checked in code). DuckDB queries are assembled from allow-listed column names with bound parameters.
- **Weights and standard errors are computed in code.** The model never averages survey data itself.
- **Too-thin cells are suppressed:** fewer than 100 people/households (30 for New York subsamples) returns "too thin to report", not a number.
- **Comparability rules live in the tools.** For example, meal detail is refused after 2021, the 2022 food-away recode is flagged, and the ATUS 2020 and October 2025 gaps are noted.
- **Every result carries its caveats**, e.g. "DCPC groceries include pharmacies", "no dollars in NHANES".

**The MTA tool** is the only tool that reaches the internet and the only one that sends a query language (SoQL), so it gets extra limits (`mta.py`):
- The model picks a station set, year, season and hour window from fixed lists. Code builds the SoQL from constants: station IDs are integers from a table, dates are computed, and no model text enters the query.
- GET only, to `data.ny.gov` only, to three allow-listed dataset IDs. Redirects are off.
- Server-side aggregation with `$limit`, a 45 s timeout and one retry, a 200 KB response cap, and a schema check on every row.
- Results are cached for 12 hours; at most 6 MTA calls per run.

**Claims**
- **Evidence or nothing.** Every query result gets an evidence ID. The Writer must end every sentence with a number in `[E#]`. Code then checks every cited ID exists and every number in a cited sentence appears in that evidence, allowing rounding and index-to-percent conversion. On a failure the Writer gets one rewrite listing the exact problems; anything still unmatched is flagged at the bottom of the report. A Sources section lists every cited query.
- **Calibrated difference language.** Every comparison in a tool's output is labeled "clear", "suggestive" or "no clear difference". The Writer must use matching words ("higher", "leans higher", "about the same"), may call a day a peak or low only when its label is clear, and may describe a weekly shape only for the sources whose labels show it.
- **Peaks and lows are checked in code.** When the report says a day is the low or peak "at" a number, `check_extremes()` finds that number's table in the cited evidence and checks the day really is the lowest or highest (of all five weekdays, or of Monday–Thursday). A mismatch goes into the rewrite request.
- **Friday is expected.** The Writer states the Friday peak briefly, explains it, and spends most of the report comparing Monday–Thursday.
- **No causal claims:** "consistent with", never "causes", "drives" or "because of". Code scans the report for causal phrases, raw data labels (e.g. `age_65_plus`) and length over 1,100 words, and includes them in the one rewrite request.
- **Places kept apart:** the NY metro (CE), NY State (DCPC) and NYC (MTA, ATUS) are named for what they are.
- **Required sections:** a "What explains it?" verdict table (Supported / Mixed / Not supported) and a "What the data can't tell us" section. The table lists only explanations a Researcher tested that bear on the question; a code filter drops any "Not tested" or "Can't tell" row, and things no dataset measures (actual paydays, each person's work location) go in the limits section. Supported needs clear evidence and nothing clear against it; suggestive or conflicting evidence is Mixed. The explanations the Lead is told to test: office days vs home days, an end-of-week habit, cooking less as the week goes on, income, payday timing and a Monday grocery restock.

**Budget** (in code): 3 research rounds, 8 Researchers, 5 per round, 6 data queries per Researcher, 6 MTA calls per run, Lead `max_turns=14`, Researcher `max_turns=10`, an 8-minute run timeout, 90 s model timeouts with 2 retries. Closing the page cancels the run.

### Tuning the "is this weekday difference real?" rule

The first design only let the report say "higher" or "lower" when a difference exceeded 2 standard errors. `calibrate_significance.py` tests rules like that against the data itself, using every "this weekday vs the other four" comparison the tools make:

- **Replication:** split each source into independent halves (CE and ATUS: alternate years; DCPC: random halves of people). For comparisons in each |z| band in one half, how often does the other half point the same way?
- **Noise:** shuffle the weekday labels so there's no weekday effect, then count how often each threshold still fires.

Before any correction, the noise test showed the CE replicate-weight SEs were about 1.3× too small: |z| ≥ 2 fired 12% of the time on shuffled data instead of ~5%. ATUS SEs, which ignore the survey design, were 1.1× too small. Those factors are now applied (`SE_SCALE` in `data_tools.py`). After that:

| |z| band | CE: same direction in other half | ATUS | DCPC | False alarms on shuffled data (CE / ATUS / DCPC) |
|---|---|---|---|---|
| 1–1.5 | 85% | 70% | 71% | ≥1: 29% / 29% / 31% |
| **1.5–2** | **100%** | **75%** | **79%** | ≥1.5: 14% / 12% / 8% |
| **≥ 2** | **98–100%** | **88–93%** | 67–75% | ≥2: 6% / 4% / 3% |

The resulting rule:
- **|z| ≥ 2 → "clear"**: replicates ~90–100% of the time, false alarm ~5%.
- **1.5 ≤ |z| < 2 → "suggestive"**: worth reporting with hedged words, since it replicates 75–100% of the time against 50% for a coin flip.
- **|z| < 1.5 → "no clear difference"**: the 1–1.5 band fires on pure noise ~30% of the time.

The DCPC replicated weakly at every level because each half is small, so its tool output tells Researchers to treat DCPC-only results as suggestive unless the CE Diary agrees. Rerun with `uv run calibrate_significance.py`, or add `--raw` to see the uncorrected numbers.

## API

`POST /chat` keeps Project 1's shape:

```json
{"message": "Which weekday do Americans spend the most on eating out?", "session_id": null}
→ {"response": "## The short answer ...", "session_id": "…",
   "tool_calls": [{"agent": "Lead", "name": "research", "args": {...}, "result": "..."},
                  {"agent": "Researcher 2", "name": "ce_diary_spending", "args": {...}, "result": "Evidence ID: E3 ..."},
                  ...]}
```

Every tool call by every agent is in `tool_calls`, with an `agent` field: the Lead, each numbered Researcher, and the Writer's handoff. When the Lead asks for clarification, `response` holds the question and suggestions. Send the user's answer with the same `session_id` to continue.

`GET /run?question=…&session_id=…` runs the same research as Server-Sent Events for the web page. `GET /health` reports the model and whether the database is present.

## Run it

```bash
gcloud auth application-default login
uv sync
uv run app.py              # http://localhost:8000
uv run --group dev pytest -q   # offline tests: data tools, guardrails, citation checker, /chat shape (no Gemini calls)
```

### Deploy

Cloud Run with continuous deployment from this GitHub repo (Developer Connect, buildpack, no Dockerfile), project `ieor-4570-f26-ek1`. Entrypoint:

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Settings: 2 GiB memory, 600 s request timeout, max 1 instance, unauthenticated access. Every push to `main` redeploys.
- **Conversation memory:** like SafeSwap, each session's history lives in server memory. The Lead gets the last 4 turns (question + reply, without the Sources list or old evidence IDs, each cut to 6,000 characters) before the new question, and starts its brief with a standalone version of the question. The Writer only sees the earlier questions, so it can't cite numbers from an earlier report. A reply to a clarifying question is just the next turn.
- **One instance:** sessions and the context panel live in memory, so they reset when Cloud Run starts a new instance (the page then says it has no earlier context).
- **600 s:** the Lead gets 480 s. If it runs out, the Writer gets up to 90 s to report from the evidence gathered so far.
- **No environment variables needed:** Vertex AI uses the service's own project credentials. `.env.example` lists the optional ones.

`data/food_week.duckdb` (38 MB) is committed because the build uses only what's in the repo. The raw survey files are not; `build_data.py` rebuilds the database from them (see "The data").

## Example run

*"Why do Americans eat out more on some weekdays than others?"* used 2 rounds, 5 Researchers and 23 evidence records.
- **Round 1** ran 4 Researchers in parallel: the baseline weekday pattern (CE, DCPC); office vs home days with the NYC office week (ATUS, MTA, office reports); cooking less as the week goes on (ATUS, NHANES); and end-of-week habit, income and payday timing (DCPC, CE, ATUS, both payday tools).
- **Round 2** tested the Monday grocery restock (CE, DCPC).
- **The report** led with Monday–Thursday: eating out is lowest on Tuesday ($7.19 per household per day, 2022–2025) and climbs to $8.16 on Thursday before Friday's $11.12. Groceries run the other way, higher on Monday ($15.68) and lowest on Wednesday ($14.19).
- **The verdicts:** end-of-week habit "Supported" (retirees and households with no one working show the Friday peak), Monday restock "Supported", office days "Mixed", cooking less "Mixed", payday timing "Not supported" (no difference on likely paydays in either the time-use survey or the payment diary).

## Files

| File | What it is |
|---|---|
| `app.py` | Agents, Lead/Researcher/Writer tools, budget, citation checker, `/chat`, `/run` |
| `data_tools.py` | Fixed queries per dataset, standard errors, labeled comparisons, evidence records |
| `mta.py` | Live MTA ridership with its own guardrails |
| `catalog.py` | Dataset descriptions the agents read; cited reference notes |
| `build_data.py` | Offline ETL: `../local data/` → `data/food_week.duckdb` |
| `calibrate_significance.py` | The replication and noise tests behind the "clear/suggestive" rule |
| `context_panel.py`, `context-panel.js` | The class context panel (unchanged) |
| `index.html` | The front end |
| `tests/test_app.py` | Offline tests |
