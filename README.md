# Food Week: deep research on when Americans spend on food

**When during the work week (Monday–Friday) do Americans spend money on food (groceries and eating out), and why?** Ask a question and a team of agents answers it from five US datasets, citing every number to the query that produced it. New York City is the case study.

**Who it's for:** café and restaurant owners planning staff and promotions by weekday, reporters covering hybrid work, and anyone curious about everyday spending.

**How to use it:** type a question about Monday–Friday food spending (or click an example) and press Research. A run takes about 2–4 minutes and always answers within 5. The activity list and the context panel show each agent working. Vague questions get a follow-up question with suggested rewrites. Follow-ups build on the conversation: once there's an earlier turn, the question box shows what the model will see, and **Start over** clears it.

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
              │  research() × 2-5 in ONE turn  ─▶  Researchers run in parallel, fresh contexts,
              │                                    each with only its assigned datasets' tools
              │◀── findings with evidence IDs [E1], [E2] …
              │
              │  gap or conflict? ─▶ one follow-up round (2 rounds at most)
              │  enough?          ─▶ write_report(brief)
              ▼
            WRITER (question + brief + findings + evidence) ─▶ code checks citations ─▶ report + Sources
```

Built on the class Wikipedia deep-research app (same Agents SDK + LiteLLM/Vertex setup and context panel), with these changes:
- **An open agentic loop.** Each turn the Lead decides whether to clarify, start parallel Researchers, start a follow-up round, or hand off with `write_report`. If it stops without handing off, code calls the Writer anyway.
- **Rounds are capped at 2, in code** (`admit_researcher()`). All `research` calls in one Lead turn count as one round.
- **One Researcher type, many copies.** Each `research` call clones it with a number (its own context-panel column) and **only the tools for the datasets the Lead assigned**.

| Agent | Model | Tools |
|---|---|---|
| **Lead** | `gemini-3.5-flash` | `list_sources` · `research(task, sources)` · `ask_user(question, suggestions)` · `write_report(brief)` |
| **Researcher** (parallel) | `gemini-3.5-flash-lite` | `describe_source` plus, for its datasets only: `ce_diary_spending` · `dcpc_spending`, `dcpc_payday_window` · `nhanes_eating` · `atus_time_use`, `atus_payday_window` · `mta_subway_ridership` (live) · `reference_note` |
| **Writer** | `gemini-3.5-flash` | none: sees only the question, the brief, the findings and the evidence |

Each data tool reads **one** dataset and is named after it. The payday tools compare days that often follow a payday (1st–3rd, 15th–17th, 28th–31st) with other days on the same weekday; no survey records actual paydays.

## The data

| Source | What it measures | Years | Where |
|---|---|---|---|
| **BLS Consumer Expenditure Survey Diary** | Household food **dollars** by weekday: groceries, fast food, full service; lunch/dinner through 2021 | 2015–2025 | US, regions, NY metro |
| **Atlanta Fed Diary of Consumer Payment Choice** | Individual food **dollars** by day, by employment and retirement | Oct 2021–2025 | US, NY State |
| **CDC NHANES** | Restaurant vs store food by weekday; **no dollars** | 2017–18, 2021–23 | US |
| **BLS American Time Use Survey** | Shopping, eating out and cooking by **where people worked that day**; **no dollars** | 2015–2025 | US, NY metro, NYC (thin) |
| **MTA Subway Hourly Ridership** (live API) | Entries by weekday at 11 Manhattan office stations or systemwide | 2017–now | NYC |

Office-attendance figures (Placer.ai, Kastle, Partnership for NYC) are hand-copied, cited notes, not queried data. Kastle's terms forbid automated access.

**Offline build.** `build_data.py` turns ~2 GB of raw SAS/XPT/CSV/fixed-width files in `../local data/` into one 38 MB DuckDB file (`data/food_week.duckdb`, committed). It handles the error-prone survey work once, in code: linking purchases to weekdays and spending codes to food categories, weights and replicate weights, NHANES design-based SEs, and ATUS work location. Build checks: CE grocery dollars match BLS's own totals within 1.4%. CE eating out excludes catered affairs (~15% of BLS food away from home). Every CE diary week covers each weekday exactly once (879,998 day rows = 125,714 weeks × 7).

```bash
uv run --group build build_data.py          # add --fresh to ignore cached intermediate tables
```

## Guardrails

**Input:** off-topic questions get a two-sentence refusal with examples. Vague ones get `ask_user` (never twice, and not when a sensible reading exists). Weekends are out of scope. Questions are cut at 2,000 characters.

**Tools: no model-written SQL.** Every argument is an enum or a bounded number, re-checked in code, and queries use allow-listed columns with bound parameters. Weights and standard errors are computed in code. Cells under 100 people (30 for New York subsamples) return "too thin to report". Comparability rules live in the tools (e.g. no meal detail after 2021, the 2022 food-away recode, ATUS gaps in 2020 and October 2025), and every result carries its caveats.

**MTA tool** (`mta.py`), the only internet access: the model picks from fixed lists, and code builds the SoQL from constants. It is GET-only to three allow-listed `data.ny.gov` datasets with no redirects, a 45 s timeout, a 200 KB cap and a schema check on every row. Results are cached for 12 h, with at most 6 calls per run.

**Claims**
- **Evidence or nothing.** Every query gets an evidence ID, and every sentence with a number must cite one. Code checks that each cited ID exists and each number appears in it (allowing rounding), plus wording (causal phrases, raw labels like `age_65_plus`, over 1,100 words). Problems go back to the Writer for one rewrite; anything still unmatched is flagged under the report.
- **Calibrated language.** Every comparison is labeled "clear", "suggestive" or "no clear difference" (see below), and the Writer must match it ("higher" / "leans higher" / "about the same"). When a survey's numbers visibly differ but aren't a clear difference, the Writer gives both and says the sample is too small to tell (prompt rule).
- **Peaks and lows are checked in code** (`check_extremes()`): a claimed low or peak "at" a number must really be the lowest or highest in the cited table.
- **No causal claims** ("consistent with", never "causes"). Places are kept apart: NY metro (CE), NY State (DCPC), NYC (MTA, ATUS). Friday's peak is stated briefly, and the report focuses on Monday–Thursday.
- **Required sections:** a "What explains it?" table (Supported / Mixed / Not supported) listing only explanations a Researcher tested, and a "What the data can't tell us" section. The Lead tests office vs home days, an end-of-week habit, cooking less, income, payday timing and a Monday grocery restock.

**Budget (in code):** 2 rounds, 8 Researchers (5 per round), 6 data queries per Researcher, 6 MTA calls per run, Lead `max_turns=14`, Researcher `max_turns=10`. Closing the page cancels the run.

**Time: answers within 5 minutes.** A run gets 270 s in total (`RUN_BUDGET`), leaving margin for network and a cold start. Research is cut at 150 s; it usually takes 45–80 s. The Writer gets the rest: its draft takes 35–90 s, and the citation-fix rewrite only starts with ≥ 45 s left. If the rewrite fails or runs late, the first draft stands, with flags. If research fails or runs out of time, the Writer reports from the findings so far. Two LiteLLM details matter here:
- **429s:** Vertex rate-limits bursts of parallel Researchers. `num_retries=5` gives exponential backoff of 1, 2, 4 and 8 s.
- **Timeouts:** LiteLLM retries a timed-out call from scratch. So the Writer has no per-call timeout (a 90 s one made slow drafts restart, taking 225 s), and the run deadline stops it instead. Lead and Researcher calls keep 90 s.

Measured locally on Nov 9, with two questions at once plus a follow-up: 115 s, 122 s and 206 s, with no 429s.

### Tuning the "is this weekday difference real?" rule

`calibrate_significance.py` tests significance thresholds against the data across every "this weekday vs the other four" comparison:
- **Replication:** split each source into independent halves. How often does the other half point the same way?
- **Noise:** shuffle weekday labels. How often does each threshold still fire?

The noise test showed CE replicate-weight SEs were ~1.3× too small (|z| ≥ 2 fired 12% of the time on shuffled data) and ATUS SEs ~1.1×. Those factors are applied (`SE_SCALE` in `data_tools.py`). After correction:

| |z| band | Same direction in other half (CE / ATUS / DCPC) | False alarms on shuffled data (CE / ATUS / DCPC) |
|---|---|---|
| 1–1.5 | 85% / 70% / 71% | ≥1: 29% / 29% / 31% |
| **1.5–2** | **100% / 75% / 79%** | ≥1.5: 14% / 12% / 8% |
| **≥ 2** | **98–100% / 88–93% / 67–75%** | ≥2: 6% / 4% / 3% |

So **|z| ≥ 2 is "clear"**, **1.5–2 is "suggestive"** (replicates 75–100% of the time vs 50% for a coin flip), and **below 1.5 is "no clear difference"**. DCPC replicates weakly because each half is small, so DCPC-only results are treated as suggestive unless the CE Diary agrees. To rerun: `uv run calibrate_significance.py` (`--raw` for uncorrected).

## API

```json
POST /chat  {"message": "Which weekday do Americans spend the most on eating out?", "session_id": null}
→ {"response": "## The short answer ...", "session_id": "…",
   "tool_calls": [{"agent": "Lead", "name": "research", "args": {...}, "result": "..."},
                  {"agent": "Researcher 2", "name": "ce_diary_spending", "args": {...}, "result": "Evidence ID: E3 ..."}]}
```

`tool_calls` lists every call by every agent. For a clarifying question, `response` holds the question and suggestions; send the answer with the same `session_id`. `GET /run?question=…&session_id=…` streams the same run as Server-Sent Events for the web page. `GET /health` reports the models and whether the database is present.

**Conversation memory:** sessions live in server memory. The Lead sees the last 4 turns (each reply without its Sources list or evidence IDs, cut to 6,000 characters) and restates the question standalone in its brief. The Writer sees only earlier questions, so it can't cite an earlier report's numbers.

## Run it

```bash
gcloud auth application-default login
uv sync
uv run app.py                  # http://localhost:8000
uv run --group dev pytest -q   # 26 offline tests: data tools, guardrails, citation checks, Writer deadline, /chat (no Gemini calls)
```

**Deploy:** Cloud Run, with continuous deployment from this repo's `main` (Developer Connect, buildpack, no Dockerfile). Project `ieor-4570-f26-ek1`, entrypoint `uvicorn app:app --host 0.0.0.0 --port $PORT`. Settings: 2 GiB memory, 600 s request timeout (a ceiling; runs end by 270 s), unauthenticated access, and **max 1 instance**. Sessions and the context panel live in memory, so one instance keeps every request on the same memory. They reset on a redeploy or when Cloud Run scales to zero after idling. No environment variables are needed (Vertex uses the service's credentials); `.env.example` lists the optional ones.

## Example run

*"Why do Americans eat out more on some weekdays than others?"* used 2 rounds, 5 Researchers and 23 evidence records. Round 1 ran 4 Researchers in parallel (the baseline pattern; office vs home days with the NYC office week; cooking less; end-of-week habit, income and payday). Round 2 tested the Monday grocery restock. The report led with Monday–Thursday: eating out is lowest on Tuesday ($7.19 per household per day, 2022–2025) and climbs to $8.16 on Thursday before Friday's $11.12. Groceries run the other way, $15.68 on Monday and lowest on Wednesday ($14.19). Verdicts: end-of-week habit Supported (retirees show the Friday peak too), Monday restock Supported, office days Mixed, cooking less Mixed, payday timing Not supported.

## Files

| File | What it is |
|---|---|
| `app.py` | Agents, tools, budget and deadline, citation checks, `/chat`, `/run` |
| `data_tools.py` | Fixed queries per dataset, standard errors, labeled comparisons, evidence records |
| `mta.py` | Live MTA ridership with its own guardrails |
| `catalog.py` | Dataset descriptions the agents read; cited reference notes |
| `build_data.py` | Offline ETL: `../local data/` → `data/food_week.duckdb` |
| `calibrate_significance.py` | Replication and noise tests behind "clear/suggestive" |
| `context_panel.py`, `context-panel.js` | The class context panel (unchanged) |
| `index.html` | The front end |
| `tests/test_app.py` | Offline tests |
