# Changelog

All notable changes to CairnIQ.

## [2.4.0] - 2026-07-24

Development since 2.2.0 has been dominated by the **advisor layer**: making the
system's own claims provable (grounding, compliance pre-checks, an audit trail, a
regression harness) and letting it speak first (a scheduled morning brief, a
persistent alerts inbox, and two zero-LLM monitors that fire on their own).

The later half of the cycle added the other side of that: **making what already
shipped visible**. Several engines were correct, running, and invisible on a
healthy machine — a number in a JSON file nothing read on a schedule. The weekly
one-page review, the turn provenance record, and the profile-readiness
switchboard exist to turn those into something a person actually reads, including
when the honest answer is "nothing this week" or "this feature is inert because
you have not filled the store behind it".

### Added

#### Alerts & background monitoring

- **Persistent alerts inbox.** Per-profile `alerts.jsonl` store (severity `info`/`warning`/`critical`, dedup key, read state) broadcast over the existing WebSocket connection, with a macOS desktop notification for `warning` and above. Dedup keys refresh an unread record instead of duplicating it; the store is capped at 500 records. Delivery: `GET /api/alerts` (with an unread filter), `POST /api/alerts/mark_read`, an `/alerts` inbox page with its own live socket, and a nav badge. First producers are market-regime flips and action-required pulses.
- **Watch-conditions engine.** Trigger levels the advisor commits to in a Today's Priority brief or a catalyst scenario are now stored and re-checked automatically. Prompts emit a machine-readable side-channel beside the prose (stripped before display); a zero-LLM scheduler tick re-reads one cached quote per symbol every 30 minutes in market hours, and a crossed level lands in the alerts inbox restating the action the advisor pre-committed to. Firing is terminal, so a level cannot alert twice; a trigger that was already true when written is voided rather than fired. Read-only `GET /api/watch_conditions` exposes the pending set, and `POST /api/watch_conditions/{id}/cancel` retires one.
- **Intraday sentinel.** The daily Market Pulse's intraday counterpart — a zero-LLM tick that fires only on a market-*state change*: VIX and SPY-drawdown band crossings, a fresh death/golden cross, and volume spikes above 2.5× average. A per-profile state store (`intraday_sentinel_state.json`) records a signal's first observation as a silent baseline so a level that has stood all session is never re-announced, and hysteresis (band dead-zones, a 2.0× re-arm floor for volume) keeps a wobbling value from alerting repeatedly. Runs on a 6-hour cooldown — roughly twice a session.
- **As-of freshness stamps and alert gates.** `tools/freshness.py` records a timestamp at *fetch* time inside the payload, so a cache hit replays the original fetch time instead of the read time. This closes a real hole: `data_freshness` is a static label about the *source*, not the observation — a quote fetched at 09:00 and served from the 1-hour cache at 15:00 still called itself "Real-time". Two notions, because the two alert paths read different data: quotes are judged by minute-age, daily bars by whether the last bar falls in the current session. Watch-conditions refuses to fire on a quote older than 45 minutes; the intraday sentinel refuses on a bar from an earlier session. Both **skip** the record rather than muting the alert — advancing state against stale data would consume the crossing and the real one would never fire. Suppressions are counted into the run record, and fired alerts now carry their vintage ("as of 10:28 (2 min ago)").
- **Auto-escalated catalysts** now raise an alert (warning when they touch held names, info otherwise) instead of landing silently in a cache.
- **Scheduled pre-market briefs.** Today's Priority is precomputed once per profile in the 7:00–9:25 window so the morning brief is ready before the open, along with a pre-market pulse and an after-close portfolio snapshot. All scheduled work is opt-in per profile via the `SCHEDULER_ENABLED` toggle in Settings (default off).
- **Nightly portfolio-neutral funnel scan.** One broad scan per trading day after the close feeds the walk-forward signal log with the funnel's *market* selection rather than any one profile's portfolio-fit overlay — previously the log only advanced when a user happened to press Scan.
- **EDGAR events task.** Once per trading day per profile, warning/critical 8-K items and fresh insider cluster buys on held names are polled and routed into the alerts inbox.
- **Holdings event radar.** The earnings, ex-dividend and FOMC calendars existed as standalone tools and were never merged against the actual book, so the dates were known and nobody was told. `tools/event_radar.py` is one deterministic zero-LLM pass over *held* names producing a nearest-first view plus **T-3 and T-1 alerts only**, each firing once — not a daily countdown, which trains the reader to skip it. Ex-dividend is T-1 only ("own it before tomorrow"). A symbol whose provider has no date produces **no event** and is listed under `unknown` instead: "no earnings coming" and "the provider did not tell us" are different facts and only one is safe to act on. Dedup keys pin the event date and the offset, so a *rescheduled* event is a fresh alert rather than one swallowed by the old key. It reports a calendar and never a trade — a test asserts the alert body contains no trade verb.
- **The drawdown playbook, and its rungs armed.** Rules written while calm and read back when it hurts: a never-sell list, a contribution priority order, a cash deployment ladder, a rebalance drift band, and a note to your future self (`GET`/`POST /api/memory/drawdown_playbook`, edited on `/context`). **Nothing here is ever authored by the app** — no default never-sell list, no suggested band — because a rule invented by software and read back during a crash carries the full authority of a promise the user made to themselves. An unset playbook produces an alert naming the absence. Storing a ladder and reciting it only on a deep crossing still left the shallow rungs decorative, so `evaluate_deployment_ladder` now arms each rung against **peak-to-date drawdown**: the −5% level the user named delivers the action the user pre-committed to, once, at the moment it is reached. Paired with the goal projection, the useful sentence at −25% is not "stay calm" but "at this level, with contributions continuing, the goal is still funded in N% of paths".
- **Weekly one-page review.** `/review` (`GET /api/weekly_review`), assembled and delivered into the alerts inbox once in the Sunday-evening window so it is waiting on Monday. Goal headline, the week's market state, how past advice actually scored, what the advisor said in the last seven days, whether the background engines are alive, and which user-authored inputs are still blank. Three contracts: **every section always renders** — a section with nothing to say says so by name and is never omitted, because a report is the highest-risk surface for the empty-block back-fill failure this codebase has already paid for; **it never generates, only reads** (pulse from cache, no LLM, no scan, so it cannot change the state it is describing); and **it never invents a number**. Its heartbeat production count is *sections assembled*, not "interesting" findings — a quiet week is the normal outcome, and counting only eventful weeks would put a working reporter on an idle streak inside a month.
- **Daily disk housekeeping.** Nothing in the tree rotated: an instance left running accumulates tens of MB of never-truncated JSONL and hundreds of MB of LangGraph checkpoints, none of it any job's responsibility. Rotation is **copy-truncate, never rename** — launchd and the file handlers hold the inode for the life of the process, so a rename leaves every writer bound to the old file. A checkpoint is pruned only when its age is **known** (LangGraph's UUID6 ids carry a 60-bit timestamp) and only **a whole thread at a time**, so a surviving conversation never loses part of its lineage; an id that will not parse as a sane date is kept rather than deleted on a guess.

#### Advisor, risk & compliance

- **Deterministic IPS compliance pre-check.** `tools/ips_precheck.py` extracts buy-side proposed trades from a draft (explicit verb + explicit size only, with negation, avoid-list, third-party, and price-cue guards so a "$200" after *at/stop/target* is read as a price, not a size) and checks the sized ones numerically against the profile's constraints — position caps, true-sector cap via the fund-decomposition stack, dollar-at-risk from a stated stop, restricted list. Rows are three-state: computed FAILs cap the score and trigger a compliance retry, `NOT_EVALUATED` never gates, and the full table reaches the Risk Judge so it confirms computed numbers instead of estimating them.
- **Risk-limits editor, and an execution-readiness gate over it.** The compliance pre-check above is described everywhere in this project as the mandatory gate on a proposal, and wherever `risk_constraints` was `{}` it was gating nothing — because **there was no screen to fill it in from** — the store, its accessor and both consumers all shipped correct, and the only writer was an undocumented POST. Nobody declined to state their limits; nobody was given a way to. **Context › Risk Limits** is that way in: four caps, a restricted list, and each blank box naming what stays switched off while it is blank (the same sentences the readiness report uses, so the two cannot tell different stories). The second half is the distinction the store could not previously express — an axis nobody has been asked about and an axis the user deliberately left open are the same empty bytes, and only the second is an answer. A confirmation records the axes left blank at that moment (`unconstrained_ack`), so a cap deleted later reverts to an open question instead of inheriting a confirmation given about different axes. Until every blank axis is confirmed, `execution_readiness()` reports sized output as **not execution-ready** through one shared seam — the pre-check, the optimizer, the drift check, and 3.8's proposals when they land. It **blocks nothing**: the check still runs, the optimizer still solves, both still report exactly what they applied, and no cap is invented to clear the flag — refusing to produce output would turn "unstated means unconstrained" into a house default by another name. It is also kept out of the pre-check's violation list, because an unanswered axis is a gap in the profile, not a fault in the draft, and the block that reaches the judge says so explicitly: name no figure, treat it as no breach, score nothing for it.
- **Risk-verdict audit trail.** Every judge verdict persists to a per-profile `risk_verdicts.jsonl`, with `get_recent_verdicts()` as the calibration reader. Records carry the IPS pre-check result alongside the verdict.
- **Golden-set eval harness.** `agent/eval/golden_harness.py` is a fixture-backed scenario corpus — each draft engineered to trip (or deliberately *not* trip) exactly one rule, with its portfolio and quotes supplied inline, no network. Scenarios cover the failures caught in production (not-held sell, fabricated portfolio total, wrong currency label, stale price, allocation mismatch, position-cap breach) plus the mirror of each, so a regression toward *over*-flagging is caught too. Deterministic scenarios run in the pytest suite; `scripts/run_eval_harness.py --live` runs the corpus through the real LLM judge via a pure `judge_advice()` seam that never writes to the audit trail. This is the gate to run before any provider, model, or prompt change.
- **Pre-trade candidate impact preview.** `preview_candidate_impact` answers "should I add this?" by recomputing the portfolio *with* a candidate at a proposed size and reporting the delta — never a verdict. It reuses the same deterministic IPS engine that gates a live instruction, so a preview can't say "fine" where the compliance gate would later fail, and reports before/after beta, volatility, 95% CVaR, correlation-to-book, and the candidate's share of proposed volatility. The opportunity funnel attaches a compact preview to its top ≤3 High-Conviction/Exceptional picks.
- **Turn-wide data provenance.** Two contracts already described degraded data and both were per-tool (`unavailable()` says "I could not check", the freshness stamp says when a payload was really fetched). Missing was the turn-level answer to *what was the evidence behind this answer actually worth?* — so a tool could fail for a missing key, the model read past it, and the advice read exactly like advice built on complete data. `tools/provenance.py` reads the rendered tool-execution context — **the same block the judge sees**, so the summary cannot disagree with the evidence it describes — and counts live, unavailable, stale and unverified sources. A payload with no readable stamp is `unverified`, never `fresh`: absence of proof is not proof of freshness. On thin evidence the judge now **caps the confidence** of the draft rather than letting a degraded turn ship at full confidence.
- **Failed-tool substitution.** When a tool failed, the analyst nodes appended `Error executing tool: …` and moved on — often terminal inside a two-cycle budget, and the answer shipped with a Data Gap a sibling tool could have filled. `agent/tool_substitution.py` is the general form of two fixes that had already shipped per-source. It deliberately does **not** use `TOOL_RELATIONSHIPS`: that graph means *related*, not *equivalent* (its entry for `get_macro_overview` lists `get_canada_macro`, so a "first healthy sibling" rule would answer a US macro question with Canadian macro). `TOOL_SUBSTITUTES` is curated by hand, verified name-by-name against the live registry, and every entry answers the **same question** as the tool it stands in for; where no honest equivalent exists there is no entry and a Data Gap remains correct. Three guards: per-call argument compatibility (nothing is invented to make a substitution fit), never chain and never repeat, and **never silent** — the result carries a notice naming both tools, without which the judge would audit the substitute's numbers against the failed tool's name and flag source fraud, correctly.
- **Profile readiness.** Every input a shipped engine depends on that only *you* can state, in one surface (`GET /api/profile_readiness`, rendered as marks and a switchboard on `/context`): whether it is on file, and **which shipped feature is inert while it is not**. Five separate slices can ship complete and correct and then sit dark because the store behind them is empty — the drawdown playbook, the drift band, the IPS caps, the few-shot pool, the wealth goal. In each case the app is right to refuse to author the value, and in each case the cost lands as an invisible blank. This reports emptiness and names the consequence; it never authors, defaults, suggests, exemplifies or ranges a value — not as a placeholder, not as an "e.g.", not as a "most people" — and a test scans every string it emits to keep it that way. It is deliberately not a nag: a blank is a valid answer, nothing is scored or chased, and the switchboard counts **features switched off**, not sentences of prose.
- **Estimation layer completed.** Return series are now converted into the profile's base currency before computing risk, so FX risk actually enters volatility, VaR, drawdown, beta and correlation for mixed-currency books (new `tools/fx_utils.py`); covariance estimation gained Ledoit-Wolf shrinkage (default), EWMA, and sample estimators in `tools/covariance.py`, implemented on numpy rather than adding scikit-learn; plus CVaR and CAGR. An unfetchable FX pair falls back to native-currency returns and says so via `data_warning` rather than silently pretending FX is neutral.

#### Data & scanning

- **SEC EDGAR pipeline.** Keyless, fair-use-throttled filings data (`tools/sec_edgar.py`) with three signals: Form 4 with *correct* transaction coding (only code P is an open-market buy, only S a sale; grants, exercises and withholding are classified as compensation mechanics — the exact miscoding in the yfinance insider table), including per-owner aggregates, a 10b5-1 flag and cluster-buy detection; 8-K material events with per-item severity (bankruptcy, restatement, auditor change, delisting → critical) via the new `get_material_events`; and 13F quarter-over-quarter diffs for a curated set of long-horizon managers via the new `get_institutional_moves`. `get_insider_activity` is now EDGAR-first with a yfinance fallback, and the 13F universe replaces the scraped media/guru feed as the scanner's institutional universe producer.
- **Bank of Canada Valet feed.** Keyless first-class CAD macro (`tools/boc_valet.py`), replacing FRED's lagged OECD re-publication of Canadian data: the policy rate with the *date and size of its last move* (a step function's level alone under-describes it), CORRA and its spread to the target (the CAD money-market stress gauge, which FRED has no equivalent for), CPI-trim and CPI-median — the core measures the BoC's own rate statements cite — against the 2% target and 1–3% band, and posted chartered-bank prime, mortgage and GIC rates. Two new agent tools, `get_canada_macro` and `get_boc_vs_fed`, the latter reporting the policy spread in basis points with the carry mechanism that transmits it into USD/CAD. `construct_bond_ladder`'s CAD path now rides the posted 1/3/5-year GIC curve (2- and 4-year rungs interpolated and flagged) instead of a flat policy-rate proxy, and `analyze_macro_context`'s Canadian block reads the BoC first and *names* which source it read. Every series carries a freshness window sized to its own publication cadence — 7 days for a business-daily rate, 95 for a monthly CPI dated the first of its reference month — and an observation past that window comes back as explicitly unavailable with its number moved to `last_value`, so a consumer reading `["value"]` gets nothing rather than a stale figure wearing a current label. The divergence tool refuses to compute at all when the US leg is FRED's hardcoded no-key fallback, rather than differencing a live rate against a literal.

- **Canadian insider filings (SEDI vocabulary).** Yahoo's insider table serves both venues with two disjoint vocabularies: US rows carry Form 4 wording ("Purchase", "Sale", "Stock Award(Grant)"), TSX/TSXV rows carry SEDI wording ("Acquisition in the public market", "Redemption, retraction, cancelation, repurchase"). A substring test for purchase/sale read a Canadian table as noise or, worse, backwards — the single most common TSX row is an **issuer buyback** whose description contains "repurchase", and "Disposition under a purchase/ownership plan" is a *sale* containing "purchase". Classification is now an ordered, most-specific-first rule table separating conviction (open-market buys and sells) from mechanics (grants, option exercises, ownership-plan accruals, gifts) and from issuer buybacks, with only conviction rows driving the signal — the same standard `sec_edgar.py` applies to Form 4 codes. Position matching is exact, never substring: bare "Issuer" *is* the company filing about its own buyback, while "Director of Issuer" is a real insider. A `.TO`/`.V`/`.CN`/`.NE` suffix now settles the venue offline instead of asking EDGAR, which holds no CIK for these names and returned a "not a US filer" note that read to the model like "no insider data exists".
- **Earnings-call transcript intelligence.** Three problems, in the order they had to be fixed. **(1) It has to know when it does not know.** The old guard tested for an error string, but a rate-limited transcript returns a *truthy* "⚠️ API Limit Reached … here is a web summary instead" — so the word counter scored the search snippet and returned a tone verdict for a call whose transcript was never read. Detection is now positive (`is_real_transcript`) and the no-data path returns `unavailable()`, which means the degradation is counted by the turn-provenance summary and can be substituted for. **(2) A twelve-word lexicon is not a signal.** The categories are now drawn from the Loughran-McDonald financial sentiment dictionary — explicitly a subset of a few hundred terms that actually recur in earnings calls, so absolute counts are stated as *lower bounds* rather than implying full coverage. **(3) The delta is the signal, not the level.** New `compare_management_tone_qoq` reports per-1,000-word shifts (cautious, confident, hedging, legal, obligation) against the same team's previous call; normalising per 1,000 words stops a longer transcript reading as a more negative one, and a consistent undercount cancels in the difference. The old interpretation strings claiming confident language "historically precedes upward earnings revisions" were removed — nothing here measured that.
- **Fund flow recorder (accruing).** Measured before a line was written: `get_shares_full` returns `None` for the *entire fund class* — every fund and ETF control tested, US and TSX alike (an equity control returns rows, so the accessor works; it is a fund-class gap, not a Canadian one), and no vendor sells the history on the plans in use. A dated daily *point* does exist, so `tools/fund_flows.py` records one row per fund per day into `user_data/fund_shares_history.csv` and accrues the series locally. **It cannot answer on the day it ships and says so**: the first week-over-week delta lands ~7 days in, a credible price-vs-flow divergence read takes 2–4 weeks, and until then `get_flow_series` reports `status="accruing"` with the day count rather than drawing a confident 0.0%. Shares rather than AUM, so no FX or NAV return enters the flow arithmetic and a CAD-listed fund is directly comparable with a USD-listed one. **One source per series**: FMP and Yahoo disagree on SPY's share count by ~15%, so every row records its source and the series refuses to difference across a change instead of manufacturing a phantom creation event.

#### Memory & personalization

- **Behavioural observation log, and a human gate before anything reaches a prompt.** The store this replaces was measurably near-mute, not noisy: days of dense use would fill the summary store to its cap while yielding a couple of durable facts. The defect was the *seam* — extraction fired on the first supervisor pass, before any tool ran and before the answer existed, so it judged one isolated message with no conversation, no answer and no outcome. `tools/observations.py` now records deterministic **post-turn** behavioural observations (regex cues, ticker shapes, a shares comparison — no LLM at this seam), and the store is **prompt-invisible**: the only path to the model runs through consolidation → `pending_lessons.json` → a human clicking confirm, enforced structurally by `tests/test_observation_invisibility.py` rather than by convention. The consolidation pass (`GET /api/observations`, `POST /api/observations/consolidate`, daily on the scheduler) is gated on a minimum evidence count, requires every candidate rule to **cite at least two observation ids from the batch it was shown** (a candidate citing an id outside the batch is dropped, never repaired), and drafts at most three per pass so one run cannot fill the queue and turn the confirm gate into a thing nobody reads.
- **Instruction cap raised to 15, and eviction is announced.** `lessons_learned` is injected into every prompt and capped; the cap is now 15 with FIFO truncation, and a write that evicts an older instruction **says which one it retired** instead of dropping it silently. Silent eviction was the bug — eviction itself is fine.

#### Portfolio, UI & platform

- **Constrained optimizer + drift-band rebalancing.** `optimize_portfolio` runs SLSQP max-Sharpe / min-vol / target-vol under the profile's **own** IPS constraints (single-name and fund caps, the fund-decomposed sector cap, the restricted list), on the Ledoit-Wolf covariance from the estimation layer. Expected returns are labelled `basis: "historical_mean"` — a description of this portfolio over a past window, not a forecast — and there is no default risk-free rate: it is 0 unless passed, and the value used is reported. `check_rebalance_drift` answers "should I rebalance?" against the user's **own** `rebalance_drift_pct` from the drawdown playbook; with no band stored the check is unavailable and says why, because a band the software chose still gets quoted back later as the user's. When the band is breached the trade list back to target is costed — turnover in base currency, and the realized-gain exposure of the sells in **taxable accounts only** (TFSA/RRSP/IRA/DCPP/PENSION realize nothing). The tax *bill* is withheld: no marginal rate is stated anywhere in the profile, and one computed from an invented rate is exactly the kind of number this codebase does not print.
- **The `/memory` page folded into `/context`** as three tabs — Your Profile, What I've Learned, and the Knowledge Graph — with the wealth goal, drawdown playbook and readiness switchboard on the first tab. Two pages that read the same store had drifted apart.
- **Advisor Ledger.** A past-recommendations scorecard at `/recommendations` (`GET /api/recommendations`) that scores prior calls against outcomes, plus a ledger menu in the nav. Recommendation restatements collapse into the original call and trimming is maturity-aware, so a ledger entry survives long enough to be scored.
- **Portfolio editor upgrades.** Last price, computed return, total value and day-move direction for manual entries, with sortable columns; broker-synced rows show the day move; manual prices survive the editor save round-trip, and a live quote beats a pinned CSV price.
- **Live run panel.** The run spinner and the reasoning box merged into one live activity log that captures reasoning from every node that streams, with a status heartbeat and a compact run dock pinned at the bottom of the chat when the trace scrolls off screen.
- **Multi-user auth.** Self-service registration and a login page, per-profile broker credentials, a public health probe, and the API endpoints the CairnIQ iOS companion client consumes.
- **Google Vertex AI (Gemini) provider**, including a keychain-stored service-account key, plus runtime LLM provider/model switching from Settings without a server restart.
- **LLM runaway protection**: instance guard, persistent spend budget, and a watchdog.

### Changed

- **Risk limits now come only from the profile.** The advisor used to cite "your 2% risk limit" and "your 10% concentration cap" back at the user; neither was ever in any profile — both were hardcoded in six places, and the judge then enforced the phantom caps and attributed them to the user. Limits now live solely in `risk_constraints` in the profile's `user_memory.json` (`GET`/`POST /api/memory/risk_constraints`). **There are no house defaults**: an unstated limit enforces nothing and is never quoted back, and results carry a `risk_basis` so a figure is never misattributed.
- **Today's Priority splits held from watching.** A BUY thesis on a name you don't own is an entry plan, not an open position. Watched names get `ENTRY TRIGGERED` / `ENTRY MISSED` / `SETUP BROKEN PRE-ENTRY`; held names keep `TARGET REACHED` / `STOP BREACHED` / `CONTRADICTED`. Previously exit flags fired on never-opened positions and the brief rendered them as sell directives.
- Integrated keyless TradingView Screener API into Opportunity Scanner Stage 0 (Dynamic Universe Assembly) for dual US (NYSE/NASDAQ) and Canadian (TSX/TSX-V) equities across all 11 GICS sectors; retired legacy static stock universe (`stock_universe.json`) in favor of 100% live dynamic market discovery.
- Tool output is compressed head+tail (reversibly) before entering context, and the tool surface was consolidated to keep expensive tools narrow for focused nodes.
- **One source of truth for the version label.** The number was typed out in four places — sidebar, Settings footer, `/api/health`, README badge — and all four had drifted to 2.2.0 while the shipped release was 2.4.0. It now resolves from `agent/version.py` everywhere, baked in rather than read from git so a zip download or packaged install still labels itself correctly. Cutting a release is `git tag vX.Y.Z` **and** bumping the constant; `tests/test_app_version.py` fails the suite if they disagree, or if the README badge does.
- Log rotation's size cap was raised to 100 MB. It is a **backstop**, not the normal path: daily rotation means nothing should ever reach it.
- Server hot-reload is off by default (opt in via `CAIRNIQ_RELOAD`).
- Dependency bumps: `langchain>=1.3.10`, `langchain-aws>=1.6.0`, `lxml>=6.1.1`, `pyarrow>=24.0.0` (`requirements.txt`), `pre-commit>=4.6.0` (`requirements-dev.txt`); CI `actions/checkout` v6 → v7.
- Further dependency bumps: `uvicorn>=0.49.0`, `PyJWT>=2.13.0`, `langchain-anthropic>=1.4.6`, `tqdm>=4.68.3`, `Jinja2>=3.1.6` (`requirements.txt`) — all pure-Python, no new system deps or Python-range change.
- `pyarrow>=24` publishes wheels for **Python 3.11–3.13 only** (drops 3.10) — corrected the stale "3.10–3.13" note in `requirements.txt`. The supported range is unchanged at 3.11–3.13.
- Docker image now installs `libxml2-dev`/`libxslt1-dev` so an `lxml>=6` source build succeeds when no wheel matches; added OS build-prereq + troubleshooting notes to the install docs.

### Fixed

- **Grounding: empty ledgers can no longer be back-filled.** A brief invented past rotation targets and an active thesis out of ledgers that were completely empty — truthiness-gated blocks emitted nothing when empty, and the silence got filled in with plausible-sounding history. Recommendation and thesis history blocks now state explicitly that they are empty.
- **Phantom compliance flags.** The pre-check read "Enforce the IPS Limit" as a proposed *sell of IPS* and "deploy elsewhere" as a *buy of MSFT*, capping clean drafts; the judge then explained the parser misfire to the user in the visible verdict. Five phantom flags in a single turn were traced and fixed.
- **Judge rule fabrication.** The judge invented a profile rule that did not exist ("trade values in the currency traded") and the advisor obeyed it in its revision. The SOURCE FRAUD rule now covers fabricated *rules*, not just fabricated numbers.
- **Cost-basis claims are verifiable.** The verification brief dropped purchase price and gain/loss, so every true drawdown claim was unverifiable by construction and got flagged as fraud. It now prints cost basis, current price and return per holding, labelled native vs base currency.
- **End-of-day quotes are labelled as such.** Alpha Vantage `GLOBAL_QUOTE` returns the prior close on some key tiers; `get_quote` now annotates the payload with `is_stale`/`as_of`/`staleness_note` instead of presenting it as a live tick.
- **A single NaN sector return 500'd `/api/market-pulse` for a whole day.** `float("nan")` does not raise, `json.dumps` writes a bare `NaN` token, and the parser downstream rejects it — and because the pulse is cached to a *daily* file, the poisoned payload was only replaced by the date rollover, so a restart could not clear it. Guarded at all four points (producer, parser, cache write, cache read); the UI now omits a sector with no usable return rather than drawing it at +0.0%, which is a claim.
- **Every profile's conversations were pooling into one store.** The LangGraph checkpointer is built in the app lifespan, before any profile is bound, so every thread had been landing in a single `_unbound` SQLite file since June. Fixed with a routing saver that resolves the store per profile at call time; the warning that should have said so was going to a logger with nowhere to write, which is why nobody saw it for a month.
- **Ledger statistics counted rows, not calls.** A restated recommendation writes several rows, so the partial-hold stats were reporting a sample more than twice its true size. Statistics now group by `(ticker, action, supersession event)`. Separately, a call that was superseded before its horizon elapsed used to retire unscored; it is now graded **at supersession**, which recovered stranded legs the ledger had silently dropped.
- **Tool-RAG was retrieving against mostly-dead edges.** Of 154 relationship edges only 55 had both ends registered — the retriever skipped the dead names silently, so one-hop expansion had been quietly degraded with nothing ever complaining. Repointed at names the registry actually knows (46 keys, 111 live edges).
- **A rate-limited transcript now reaches the web-search fallback** instead of returning a truthy limit notice that downstream code read as content, and Tavily quota exhaustion **opens a circuit** rather than re-hitting a dead plan on every call.
- **The compliance pre-check stopped accusing a correctly sized draft**, and the scanner's headwind gate stopped hiding earnings risk through an off-by-one in the proximity window.
- The nightly `VACUUM` is skipped when a checkpoint store is still being written — its exclusive lock could fail a concurrent chat turn.
- The fund-shares recorder honours `SCHEDULER_ENABLED` like every other global task; the context page's knowledge graph no longer depends on a single `ResizeObserver` firing.
- Insider *selling* is no longer read as a sell recommendation.
- IPS trade sizing is currency-correct, and shares-based candidate sizes convert to base currency.
- The judge re-audited the original draft instead of the revision on a retry pass, critical-failing compliant revisions.
- Cross-profile leaks in the deep-reasoning executors, background worker threads, and graph-memory operations.
- The scaffold stripper ate the `<watch>` side-channel before it could be captured, so the watch-conditions engine ran dead on its first morning and leaked raw JSON into the brief.
- The TradingView screener fetch now goes through `requests` (which carries its own CA bundle) — a raw urllib HTTPS call fails certificate verification on a framework Python with no system CA file, which silently zeroed the scanner's universe.
- UI: an unescaped apostrophe that killed the entire sidebar script, a dashboard 500 for profiles with an age but no income, the news feed collapsing to zero height, the reasoning trace collapsing the moment it filled, and a false init failure reported on every non-chat page.

### Internal

- New default-deny `scripts/local/` for ad-hoc / personal / maintainer scripts (gitignored except its README), replacing the per-script `.gitignore` list — keeps one-off fixes from leaking into the public repo.
- Test suite defaults `CAIRNIQ_AUTH_REQUIRED` off so a developer's real `user_data/.env` no longer 401s the API tests.
- **Fixtures are synthetic, always.** Test data is invented rather than captured from a running instance. The secret-scanning hooks are not a substitute for this and never were: they look for credentials, and a fixture carrying none will pass every one of them.
- Suite grew to ~2,640 tests across 179 files alongside the advisor work.
- Bandit is version-pinned in CI, and the gitleaks download retries. An unpinned scanner turns "a new release added a check" into a red build on a commit that changed nothing, and a TLS hiccup fetching the scanner reported as a failed security scan before anything had been scanned.
- Application icons are no longer tracked; nothing in the app or the packaging step loaded them.

## [2.2.0] - 2026-06-11

### Added
- Catalyst Engine (Layers 2–3): news headlines are classified into a ranked, two-lane catalyst list, with bounded auto-escalation that pre-computes full event→exposure→scenario reports (⚡ Scenario instant drill-down, cached server-side at no extra analysis cost)
- "Trump Yap": quick-action button + real-time Truth Social feed crawler (`get_latest_trump_yaps`) so the agent can factor political/policy statements into market and supply-chain analysis
- Opportunity Funnel V2: top-down accumulation-first broad-market scanner (dynamic universe assembly, theme ranking, additive scoring with capped flow bonus + entry-stage multiplier, surfaced risk overlay)
- Market-wide TSX movers scanner (`scan_tsx_movers`): screens the entire Toronto exchange for gainers/losers plus a large-cap most-active lane, with TSX Composite + USD/CAD context — closes the gap that made every Canadian-movers query fall back to news search
- Azure OpenAI and Google Gemini LLM providers (joining Anthropic, OpenAI, and AWS Bedrock), with vendor-neutral embeddings for tool retrieval
- Externalized scanner tuning via `user_data/funnel_config.json` — auto-seeded from `funnel_config.example.json` on install and first server startup; documented in the new [Funnel Configuration Guide](technical/FUNNEL_CONFIG.md)
- Resizable intelligence panel (drag handle, persisted width) and header run notices

### Changed
- Each ⚡ Scenario now opens in its own fresh chat instead of stacking in one thread; the button only appears when a real cached scenario exists (otherwise Analyze → runs a live drill-down)
- Funnel fast-screen cut is now deterministic (symbol tiebreak) and its technicals are daily-cached, so intraday re-runs return a stable pick list instead of drifting
- Optional guru/media-sentiment feed is fully fail-safe and surfaces `guru_enabled`, so confidence grading no longer downgrades picks just because the feed is absent
- Non-secret config is read from the `.env` file in settings to prevent stale environment overwrites

### Fixed
- Catalyst scenario cache leaked the raw LLM message repr (`content=… response_metadata=…`) instead of clean markdown
- Context-summary updates that fail (e.g. provider content-filter blocks) are now logged as failures instead of falsely logging "Updated"
- Health check is provider-aware; NaN JSON serialization and keychain fallback fixes

### Security
- Added local secret-scanning via pre-commit (gitleaks + detect-private-key) — blocks commits containing secrets before they enter git history
- Enabled GitHub secret scanning + push protection on the repository
- Documented dev tooling setup (`requirements-dev.txt`, `pre-commit install`) in `CONTRIBUTING.md`

## [2.1.0] - 2026-05-20

### Added
- Multi-provider LLM support: Anthropic, OpenAI, and AWS Bedrock selectable from Settings
- Flip-flop prevention for portfolio decisions (avoids oscillating buy/sell signals)
- Questrade and Alpaca live broker sync improvements
- Profile switching via cookie-based session management
- Origin allowlist CSRF protection on all mutating API endpoints

### Changed
- Upgraded to FastAPI with full async support across all routers
- Improved portfolio deduplication: live broker sync takes precedence over CSV uploads
- Demo mode now locks settings to prevent accidental overwrite of real configuration

### Fixed
- LangGraph checkpointing edge cases under concurrent requests
- Tool rate-limit rotation now gracefully falls back when all keys are exhausted
- `log_context_token` UnboundLocalError in frontend log endpoint

## [2.0.1] - 2026-04-08

### Added
- Custom desktop launcher with fancy ASCII art and system checks
- Custom application icons in `assets/icons/` (PNG files distributed with the repo; `.icns` macOS bundles are local-only)
- Comprehensive documentation structure (user guides, technical docs)
- Organized project structure with proper directories
- Stop button functionality to cancel running analyses
- Spinner clearing on cancellation for better UX

### Fixed
- Health check timeout issues (added 120s timeout wrapper)
- Health check deadlock with automatic lock release
- Spinner persistence after clicking stop button
- Import errors in health check code
- Cancellation handling in deep reasoning node

### Changed
- Reorganized documentation into docs/ directory
- Moved debug/fix documentation to docs/archive/
- Moved old installation scripts to scripts/archive/
- Moved Docker files to scripts/docker/
- Moved icon files to assets/icons/
- Cleaned up root directory (removed 6 unnecessary files)
- Updated .gitignore for better organization

### Removed
- Duplicate configuration files (.env.lock, requirements.freeze.txt)
- Obsolete pyproject.toml (conflicted with requirements.txt)
- Temporary debug scripts and logs
- Old package archives (.tar.gz, .zip files)
- System files (.DS_Store)

### Documentation
- Created comprehensive README.md
- Added detailed INSTALLATION.md guide
- Added complete USER_GUIDE.md
- Added TROUBLESHOOTING.md with solutions
- Added PROJECT_STRUCTURE.md overview
- Archived 30+ debug/fix documentation files

## [2.0.0] - 2026-04-06

### Added
- Multi-agent architecture with LangGraph
- Deep Reasoning node with Tree-of-Thought analysis
- Market Analyst with 90+ specialized tools
- Opportunity Scanner for market-wide analysis
- Portfolio management with Questrade integration
- Real-time market data from multiple APIs
- User memory system for personalized recommendations
- Health check system for tool diagnostics
- Modern web UI with streaming responses
- Dashboard with portfolio analytics
- News feed integration
- Thesis journal for tracking analyses

### Technical
- FastAPI web server with WebSocket support
- DSPy modules for structured outputs
- FAISS vector search for semantic tool retrieval
- AWS Bedrock integration (Claude models)
- Comprehensive logging system
- Session management and checkpointing

---

## Version History

- **2.4.0** (2026-07-24) - Alerts inbox, watch conditions, intraday sentinel, freshness gates, IPS compliance pre-check, SEC EDGAR + BoC feeds
- **2.2.0** (2026-06-11) - Catalyst Engine, Opportunity Funnel V2, TSX movers scanner, Azure/Gemini providers
- **2.1.0** (2026-05-20) - Multi-provider support, CSRF hardening, portfolio UX improvements
- **2.0.1** (2026-04-08) - Documentation cleanup and UX improvements
- **2.0.0** (2026-04-06) - Initial multi-agent release
- **1.x** - Legacy single-agent system (deprecated)

## Upgrade Notes

### From 2.2.0 to 2.4.0
- No breaking changes to the portfolio CSV format
- Re-run `./install.sh` (or `pip install -r requirements.txt`) — several dependencies moved, and `pyarrow>=24` publishes wheels for **Python 3.11–3.13 only**. The supported range is unchanged at 3.11–3.13; a 3.10 interpreter will no longer resolve
- Docker builds need `libxml2-dev`/`libxslt1-dev` for the `lxml>=6` source build; the image already installs them
- **Background work is opt-in and off by default.** Scheduled briefs, the sentinels and the nightly scans stay dormant until `SCHEDULER_ENABLED` is turned on per profile in Settings
- **Risk limits no longer have house defaults.** Any cap the advisor previously quoted was hardcoded; caps now come only from `risk_constraints` in the profile, and an unstated limit enforces nothing. Set yours under Context › Risk Limits, or accept that sized output is reported as not execution-ready
- Multi-user auth is available; a single-user install keeps working unchanged

### From 1.x to 2.0.0
- Complete rewrite with multi-agent architecture
- New installation process required
- Portfolio CSV format unchanged (compatible)
- API keys need to be reconfigured in new .env format

### From 2.1.0 to 2.2.0
- No breaking changes to portfolio CSV format or API key configuration
- Re-run `./install.sh` (or `pip install -r requirements.txt`) — yfinance >=1.4.1 is required for the TSX movers scanner
- Optional: tune the new scanner via `user_data/funnel_config.json` (auto-seeded on first startup)

### From 2.0.1 to 2.1.0
- No breaking changes to portfolio CSV format or API key configuration
- Re-run `./install.sh` to install new dependencies
- Set `LLM_PROVIDER` in Settings to switch between Anthropic, OpenAI, or Bedrock

### From 2.0.0 to 2.0.1
- No breaking changes
- Documentation reorganized (check new docs/ structure)
- Desktop launcher updated (copy new version to Desktop)
- Run `./install.sh` to update dependencies

## Future Roadmap

### Planned Features
- [ ] Mobile app (iOS/Android)
- [ ] Advanced backtesting engine
- [ ] Options trading analysis
- [ ] Crypto integration
- [ ] Multi-portfolio support
- [ ] Social trading features
- [ ] Advanced risk modeling
- [ ] Tax optimization tools

### Under Consideration
- [ ] Voice interface
- [ ] Automated trading execution
- [ ] Custom indicator builder
- [ ] Community marketplace for strategies
- [ ] Integration with more brokers

---

**Current Version**: 2.4.0
**Release Date**: July 24, 2026
**Status**: Stable

> The version label here is prose and drifts. `agent/version.py` is the single
> source of truth, and `tests/test_app_version.py` holds it to the newest release
> tag and the README badge.
