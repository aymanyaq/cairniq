# Architecture

## Runtime Shape

The application is a FastAPI server with four main layers:

1. `server.py` + `api/routers/` handle HTTP routes, NDJSON chat streaming, websocket status updates, auth, profile resolution, and session persistence.
2. `agent/` contains the LangGraph workflow, node logic, shared utilities, DSPy modules, prompts, and semantic tool retrieval.
3. `tools/` contains the domain capabilities for portfolio analysis, macro data, market data, news, screening, risk, and diagnostics (~130 registered agent tools).
4. `tools/scheduler.py` runs the background layer: an in-process task loop that produces briefs and alerts without a user in the loop.

## Agent Flow

1. `Supervisor` chooses the next specialist based on the current human turn.
2. `PortfolioManager`, `NewsAnalyst`, `MarketAnalyst`, or `DeepReasoning` handle the request.
3. `RiskManager` performs a final challenge/review pass for advice-generating flows.
4. The chat route streams visible output while persisting assistant responses and session cost.

## The Advice Gate

Anything that reads as advice passes through a review gate before it reaches the user. The gate is deliberately split between deterministic code and the LLM judge, because the two fail differently:

- **Deterministic pre-checks run first.** Grounding (does every number and holding claim trace to context the model actually had?) and the IPS compliance pre-check (`tools/ips_precheck.py`, buy-side proposed-trade extraction checked numerically against the profile's stated caps). Their results are *given* to the judge as a table, so it confirms computed numbers instead of estimating them.
- **The LLM judge audits what code can't.** Source fraud, fabricated rules, unsupported reasoning. `judge_advice()` in the risk node is a **pure seam**: it runs the pre-audit, builds the prompt, invokes the judge, and parses the verdict — nothing else. Persistence, retry, and streaming stay in the caller, which is what lets the eval harness run the real judge without polluting the audit trail.
- **Every verdict is recorded** to a per-profile `risk_verdicts.jsonl` for calibration.
- **Constraints have no house defaults.** Risk limits come only from `risk_constraints` in the profile. An unstated limit produces no pre-check row at all — not a `NOT_EVALUATED` row, which the judge reads as a magnitude miss and would use to re-invent the cap.

Two things wrap the gate rather than sitting inside it:

- **Turn provenance** (`tools/provenance.py`) reads the *rendered tool-execution context* — the same block the judge is given as grounding evidence — and summarises what the turn's evidence was worth: sources live, unavailable, stale, unverified. Reading the judge's own view is deliberate; a summary collected somewhere else could disagree with the evidence it describes, and it would disagree in the direction that hurts ("all sources live" beside a context full of unavailable payloads). Unstamped is `unverified`, never `fresh`. Thin evidence caps the draft's confidence.
- **Tool substitution** (`agent/tool_substitution.py`) runs before a failed tool becomes a Data Gap, standing in a hand-curated *equivalent* — never a merely related tool from `TOOL_RELATIONSHIPS`, which means "related" and would answer a US macro question with Canadian macro. One substitution per failed call, never chained, never repeated, and never silent: the result carries a notice naming both tools, because the rendered context header carries the *original* tool's name and the judge would otherwise audit the substitute's numbers against it and flag source fraud, correctly.

`agent/eval/golden_harness.py` is the regression gate for this layer: a fixture-backed corpus where each draft trips exactly one rule, plus the mirror of each so a regression toward over-flagging is caught too. Run `scripts/run_eval_harness.py --live` before any provider, model, or prompt change.

## Background Layer

`tools/scheduler.py` ticks every 5 minutes and runs tasks under a cooldown, a lock, and a timeout. Cooldowns are the frequency control; tasks whose real gate is internal (a market window plus a per-profile daily marker) use a short cooldown so they are actually *invoked* inside their window.

| Task | Cadence | Purpose |
| :--- | :--- | :--- |
| `exchange_rate` | 1h | FX refresh |
| `portfolio_snapshot` | after close, once/day/profile | History point |
| `score_recommendations` | 24h | Advisor Ledger scoring |
| `cache_cleanup` | 24h | Expire caches |
| `housekeeping` | 24h | Rotate logs (copy-truncate), prune aged checkpoint threads |
| `premarket_pulse` | pre-market window, once/day/profile | Market pulse |
| `priority_precompute` | 07:00–09:25, once/day/profile | Today's Priority brief, ready before the open |
| `funnel_signal_scan` | after close, once/day globally | Portfolio-neutral broad scan into the walk-forward signal log |
| `edgar_events` | once/trading day/profile | 8-K severity + Form 4 cluster buys on held names |
| `event_radar` | once/trading day/profile, morning | Earnings / ex-div / FOMC T-3 and T-1 sweep for held names (zero LLM) |
| `watch_conditions` | 30 min in market hours | Re-check advisor-authored trigger levels (zero LLM) |
| `intraday_sentinel` | 6h in market hours | Market-state change detection + deployment-ladder rungs (zero LLM) |
| `observation_consolidation` | 24h | Follow-through sweep always; one gated LLM pass per profile past *n* |
| `weekly_review` | hourly check, fires Sunday evening/profile | Assemble the one-page review and deliver it to the inbox (zero LLM) |
| `fund_shares_record` | every tick, one post-close sweep/day globally | Record fund share counts — a missed day is gone for good (no vendor sells the history) |

Scheduled work is **opt-in per profile** via the `SCHEDULER_ENABLED` setting (default off).

## Alerts Rail

`tools/alerts.py` is the single delivery path for anything the system wants to say unprompted. `raise_alert()` appends to a per-profile `alerts.jsonl`, broadcasts over the existing WebSocket connection manager (thread-safe via the captured main loop), and posts a macOS desktop notification for `warning` and above. Dedup keys refresh an unread record rather than appending a duplicate; the store is capped at 500 with atomic tmp+replace rewrites.

Producers: market-regime flips, action-required pulses, auto-escalated catalysts, EDGAR material events and cluster buys, crossed watch conditions, and intraday sentinel state changes.

Two rules keep the rail from crying wolf:

- **State changes only.** Both zero-LLM monitors record a signal's first observation as a silent baseline and alert only on a subsequent *change*, with hysteresis on band boundaries. A level that has stood all session is not news.
- **Freshness is proved, not assumed.** `tools/freshness.py` stamps a fetch time *inside* the payload, so a cache hit replays the original fetch time instead of the read time. Quotes are judged by minute-age (45-minute ceiling on the alert path), daily bars by whether the last bar falls in the current session. A record that fails its gate is **skipped, not muted** — advancing state against stale data would consume the crossing and the real one would never fire. Unstamped data is *unverified*, not stale: it still fires, labelled as such, so a missing stamp upstream can't silently switch the engines off.

## Read Surfaces

An engine that is correct, running, and invisible is indistinguishable from one that is dead. Three surfaces exist purely to be read, and all three add no intelligence of their own:

- **Weekly review** (`tools/weekly_review.py` → `/review`) — a once-a-week assembly over surfaces that already exist. Three contracts: every section always renders (an omitted section is a silence, and silence in a report gets back-filled); it reads only, never generates (pulse from cache, no LLM, no scan, so it cannot change the state it describes); and it never invents a number. It reads through the same accessor its consumer reads through rather than re-deriving from JSON, and it counts **distinct calls, not ledger rows**.
- **Profile readiness** (`tools/profile_readiness.py` → `/context`) — which user-authored inputs are blank and which shipped feature is inert as a result. It never authors, defaults, suggests, exemplifies or ranges a value; a test scans every emitted string to keep it honest. Blank is a valid answer, so nothing is scored, ranked or chased.
- **Engine health** (`GET /api/engine_health`) — per-engine liveness. The rule that makes it useful: report the count that proves the **chain** ran, never a rare event. A run record saying "coroutine completed" is also satisfied by an early return.

## Memory Tiers

Behavioural memory is three stores with a structural boundary, not a convention:

`observations.json` (deterministic per-turn records, **prompt-invisible**) → gated consolidation → `pending_lessons.json` (candidates, each citing ≥2 observation ids from its own batch) → **human confirm** → `lessons_learned` (prompt-visible, capped at 15, FIFO, and an eviction names what it retired).

No LLM runs at the per-turn seam — a one-shot judgment on each turn is the defect this design replaced. The judging happens once, later, in the gated pass, over accumulated evidence. `tests/test_observation_invisibility.py` asserts the invisibility boundary on the source rather than trusting it.

## Operational Constraints

- The app keeps active websocket connections, cancellation state, and live chat session state in memory.
- Because of that, direct local launches should stay single-worker unless state is moved to shared infrastructure.
- Profile-specific user data is resolved through `tools.user_profile.get_data_path(...)`; background tasks must bind a profile explicitly, since a `ContextVar` set per request does not carry into a scheduler tick.

## Current Design Priorities

- Keep prompts aligned with actual tool names.
- Prefer direct-data synthesis when a node already knows the critical data.
- Keep the expensive tool surface narrow for nodes with a focused job.
- Prefer deterministic code over the model for anything checkable, and give the model the computed result rather than asking it to estimate.
