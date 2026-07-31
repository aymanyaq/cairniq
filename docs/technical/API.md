# API Reference

CairnIQ exposes a REST + SSE/NDJSON API under the same host that serves the web UI (default `http://localhost:8000`). This document lists every endpoint registered under `api/routers/`.

Profile resolution happens in middleware on every request — see [Profile Resolution](#profile-resolution) below.

---

## Chat & Streaming

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Submit a chat turn; streams NDJSON status/text events back. |
| `POST` | `/api/chat/stop` | Cancels the active run for the calling thread. |
| `GET` | `/api/chats` | Lists saved chat sessions for the active profile. |
| `GET` | `/api/chats/{session_id}` | Loads a specific saved session. |
| `DELETE` | `/api/chats/{session_id}` | Deletes a saved session from disk. |

### `POST /api/chat` streaming format

Responses are newline-delimited JSON. Common envelopes:

```jsonc
{"thread_id": "…"}                        // first frame, always
{"status": "Initializing agents...",      // human-readable progress ticks
 "elapsed": 12, "heartbeat": true}        //   heartbeat frames carry elapsed seconds
{"text": "Apple's Q4 results..."}         // visible answer, re-sent cumulatively
{"thinking": "..."}                       // live reasoning trace — rendered in the run panel, not the answer
{"fatal_error": "…"}                      // terminal failure; UI shows a persistent error card
```

`{"text": …}` carries the full cleaned answer so far, not a delta. Before each send the route strips `<thinking>` blocks (complete and half-streamed), leaked prompt-scaffold tags, and the `<watch>` watch-conditions side-channel, so none of them can flash in the chat on their way to the store.

Reasoning is **no longer discarded** — it is forwarded on its own `thinking` frames and rendered in the live run panel. Whether a provider returns reasoning at all is configuration-dependent (Gemini needs `include_thoughts`), so an empty trace panel is expected on some setups. A heartbeat frame is emitted when the queue is idle, so the UI always shows progress during a long tool call.

Final cost and metadata land in the last frame.

---

## Dashboard & Portfolio

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/dashboard-data` | Aggregated portfolio summary + sector allocation for the dashboard panel. |
| `GET` | `/api/benchmark` | Portfolio vs SPY benchmark history (daily-cached). |
| `GET` | `/api/portfolio/download-template` | Streams a fresh CSV template with the supported schema. |
| `POST` | `/api/portfolio/upload` | Accepts a CSV upload and replaces `user_data/my_portfolio.csv`. Force-refreshes the portfolio cache so the next page render shows the new data. |
| `POST` | `/api/portfolio/save` | Saves edits made in the Portfolio Editor table. |

---

## Thesis Journal

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/journal` | Active theses and historical trade decisions (optionally filtered by `symbol`). |
| `POST` | `/api/journal` | Log a new thesis or trade. |
| `POST` | `/api/journal/close` | Close an active thesis, recording exit price and the lesson. |
| `DELETE` | `/api/journal/{trade_id}` | Remove a thesis or trade entry. |
| `POST` | `/api/journal/reconcile` | Auto-sync active theses against live portfolio holdings. |

---

## Feedback

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/feedback` | Attach a rating (and optionally a written correction) to one chat turn. |
| `GET` | `/api/feedback/stats` | Rating counts plus the size of the high-quality few-shot pool. |

The few-shot pool draws only from *rated* turns, so a store with a hundred recorded interactions and zero ratings supplies zero examples. `/api/feedback/stats` reports the pool size separately from the total for exactly that reason — and the profile-readiness surface names the pool, not the total, as the thing that must be non-empty for the feature to be live.

---

## Alerts

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/alerts` | Alerts for the active profile, newest first. Supports an unread-only filter. |
| `POST` | `/api/alerts/mark_read` | Marks one or more alerts read. |

Alerts are stored per profile in `alerts.jsonl` with `{id, ts, severity, title, message, source, read, dedup_key, data}`. Severity is `info`, `warning`, or `critical`; `warning` and above also raise a macOS desktop notification. A repeat of an existing `dedup_key` refreshes the unread record rather than appending a duplicate, and the store is capped at 500 records. New alerts are pushed over the WebSocket to any open page, which is what drives the nav badge and the `/alerts` inbox live.

---

## Watch Conditions

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/watch_conditions` | The pending trigger set the advisor has committed to (read-only). |
| `POST` | `/api/watch_conditions/{condition_id}/cancel` | Retires a pending condition so it can no longer fire. |

Conditions are authored by the agent through a `<watch>` side-channel in its prompts, stripped before display, and re-checked by a zero-LLM scheduler tick every 30 minutes in market hours. Firing is terminal — a level alerts at most once — and a condition that was already true when written is voided rather than fired.

---

## Advisor Intelligence

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/priority` | Today's Priority brief. Serves the pre-market precomputed brief when the scheduler has one cached, otherwise runs it live. |
| `GET` | `/api/catalysts` | Ranked catalyst list from the Catalyst Engine. |
| `GET` | `/api/catalysts/scenario/{catalyst_id}` | The pre-computed event→exposure→scenario report for an auto-escalated catalyst. |
| `GET` | `/api/recommendations` | Scored past recommendations plus performance statistics (the Advisor Ledger scorecard). |
| `GET` | `/api/weekly_review` | The assembled weekly one-page review (rendered at `/review`). |
| `GET` | `/api/goal_projection` | Monte Carlo projection of the stated wealth goal — non-depletion and goal-funded rates. |
| `GET` | `/api/profile_readiness` | Which user-authored inputs are on file, and which shipped feature is inert while each is blank. |
| `GET` | `/api/engine_health` | Per-engine liveness: last run, outcome, and the production count that proves the chain ran. |

Ledger statistics group by `(ticker, action, supersession event)` — **distinct calls, not ledger rows**. A restated recommendation writes several rows, so a row count reports a sample larger than the one that exists. A call superseded before its horizon elapses is graded at supersession rather than retiring unscored.

`/api/weekly_review` is read-only in the strict sense: it assembles from surfaces that already exist, reads the market pulse from cache rather than kicking one off, and runs no LLM. **Every section is always present**, including when its content is "nothing this week" — an omitted section is a silence, and silence in a report gets back-filled by the reader.

`/api/profile_readiness` reports emptiness and names the consequence. It never authors, defaults, suggests, exemplifies or ranges a value for any input it reports on; a test scans every string the module emits to enforce that.

---

## Settings & Secrets

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/settings/save` | Persists configuration. Secret keys route to the OS keychain via `tools.secrets_store.set_secret`; non-secret keys go to `user_data/.env`. Empty secret values are treated as no-op when an existing keychain entry is present (defensive guard). |
| `GET` | `/api/session/cost` | Current session's accumulated LLM cost in the configured base currency. |

The settings handler also auto-detects Alpaca paper vs live mode by probing both endpoints with the submitted keys and sets `ALPACA_PAPER_MODE` accordingly.

---

## Memory & Knowledge Graph

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/memory/profile` | Update the user profile (age, risk tolerance, base currency, etc.). |
| `POST` | `/api/memory/theses` | Add a new active thesis (Buy/Hold/Sell with logic and target). |
| `DELETE` | `/api/memory/theses/{thesis_id}` | Remove an active thesis. |
| `POST` | `/api/memory/lessons` | Append a lesson learned. |
| `DELETE` | `/api/memory/lessons/{index}` | Remove a lesson by its index. |
| `POST` | `/api/memory/facts` | Append a key fact about the user. |
| `DELETE` | `/api/memory/facts/{index}` | Remove a fact by index. |
| `POST` | `/api/memory/sync_from_facts` | Re-derives profile fields (age, income, retirement age) from free-text facts. |
| `POST` | `/api/memory/graph/node` | Insert a node into the knowledge graph. |
| `DELETE` | `/api/memory/graph/node/{name}` | Remove a node by display name. |
| `POST` | `/api/memory/graph/edge` | Insert an edge. |
| `DELETE` | `/api/memory/graph/edge` | Remove an edge (matched by source + target in the body). |
| `POST` | `/api/memory/extract_thesis` | Extract a structured thesis from free-text via DSPy. |
| `GET` | `/api/memory/risk_constraints` | Read the profile's stated risk limits, plus `execution_readiness`. |
| `POST` | `/api/memory/risk_constraints` | Set the profile's risk limits (position cap, fund cap, sector cap, dollar-at-risk, restricted list). `acknowledge_unconstrained: true` confirms the axes left blank by this write; `false` withdraws it. |
| `GET` | `/api/memory/financial_goal` | Read the stated wealth goal (targets, horizon, annual contribution). |
| `POST` | `/api/memory/financial_goal` | Set the wealth goal. Unset means no projection is made on the user's behalf. |
| `GET` | `/api/memory/drawdown_playbook` | Read the drawdown playbook (never-sell list, contribution priority, deployment ladder, drift band, note to self). |
| `POST` | `/api/memory/drawdown_playbook` | Store the playbook. Blank fields claim nothing — the app never authors an entry. |
| `GET` | `/api/memory/lessons/pending` | Candidate rules drafted from behavioural observations, awaiting human confirmation. |
| `POST` | `/api/memory/lessons/pending/{lesson_id}/confirm` | Promote a candidate into `lessons_learned` (the prompt-visible store). |
| `DELETE` | `/api/memory/lessons/pending/{lesson_id}` | Discard a candidate. |
| `GET` | `/api/observations` | The raw behavioural observation log for the profile. |
| `POST` | `/api/observations/consolidate` | Run the gated consolidation pass over accumulated observations. |

### Memory tiers

Three tiers, and the boundary between them is structural, not conventional:

1. `observations.json` — deterministic per-turn behavioural records (what the user *did*: acted, ignored, pushed back, repeated). **Prompt-invisible.** Nothing here is ever injected into a prompt; `tests/test_observation_invisibility.py` asserts that on the source.
2. `pending_lessons.json` — candidate rules drafted by the consolidation pass. Each must cite at least two observation ids from the batch it was shown, or it is dropped rather than repaired. At most three drafts per pass.
3. `lessons_learned` (in `user_memory.json`) — prompt-visible, capped at **15** with FIFO truncation. A write that evicts an older instruction names the one it retired. The only path from tier 2 to tier 3 is an explicit human confirm.

### Risk constraints

Risk limits live only in `risk_constraints` in the profile's `user_memory.json` and are the sole source for the IPS compliance pre-check and the Risk Judge. **There are no house defaults.** A limit that has not been set enforces nothing, is never quoted back to the user as "your limit", and produces no pre-check row at all — deliberately not a `NOT_EVALUATED` row, which the judge would read as a magnitude miss and re-introduce the phantom cap.

#### Execution readiness

An unset cap answers "what is enforced" but not "was the user ever asked" — and for a long time the answer to the second was no, because nothing wrote to this store. `execution_readiness()` (`tools/ips_precheck.py`) separates the two, and is the seam every proposal surface reports through: the IPS pre-check, the optimizer, the drift check, and 3.8's decision proposals when they land.

| Field | Meaning |
|---|---|
| `stated` | Caps the user typed. Unchanged contract — these are the only ones enforced. |
| `unconstrained_by_choice` | Axes left blank and explicitly confirmed as unlimited. |
| `unanswered` | Axes left blank that nobody has confirmed. |
| `execution_ready` | `unanswered` is empty. |
| `note` | The one shared sentence for the not-ready case, or `""`. |

The confirmation is stored as `unconstrained_ack: {acknowledged_at, axes}` inside `risk_constraints`, and records the axis **names** rather than a bare flag: clearing a cap later leaves that axis outside the confirmed set, so a deleted limit reverts to an open question instead of inheriting a confirmation given about different axes. `stated_caps()` and `load_ips_constraints()` ignore the key entirely — it is never readable as a cap.

Not execution-ready **blocks nothing**. The pre-check still runs, the optimizer still solves, and both still report exactly what they applied; the flag rides alongside. Refusing to produce output would turn "unstated means unconstrained" into a house default by another name. It is also kept out of the pre-check's `violations` list: an unanswered axis is a gap in the profile, not a fault in the draft, and scoring it would penalize advice that did nothing wrong.

Entry surface: **Context › Risk Limits**.

---

## News & Market Pulse

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/news-feed` | Latest news filtered for current holdings + general market events. |
| `GET` | `/api/market-pulse` | Current market regime (VIX, breadth, sector rotation, fear/greed). |
| `GET` | `/api/market-pulse/history` | Historical pulse readings for trend visualization. |

---

## Authentication

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/auth/register` | Self-service user registration. |
| `POST` | `/api/auth/login` | Returns a Bearer access token (for the iOS client) and sets an httponly `cairniq_token` cookie plus the legacy `profile` cookie so the browser UI keeps working. |
| `POST` | `/api/auth/logout` | Clears the token cookie. |
| `GET` | `/api/auth/me` | Whoami for the presented token or cookie. |

Auth is enforced only when `CAIRNIQ_AUTH_REQUIRED` is on. Leave it on for any host that isn't loopback-only.

---

## Health

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Public liveness probe (no auth). The iOS client pings it to validate a user-entered server URL. |

---

## Profile Management

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/profiles` | List all configured profiles. |
| `POST` | `/api/profile/switch` | Switch active profile; sets the `profile` cookie. |
| `DELETE` | `/api/profile/switch` | Clear the `profile` cookie (reverts to `ACTIVE_PROFILE` or `default`). |

### Profile Resolution

Middleware reads the active profile from, in priority order:
1. Demo mode (`DEMO_MODE=true`) pins every request to the demo profile.
2. An authenticated identity — a Bearer token (iOS) or the httponly `cairniq_token` cookie (web UI). A valid token binds the profile from its claim and **ignores** the unauthenticated `profile` cookie.
3. `profile` cookie (legacy single-user path).
4. `ACTIVE_PROFILE` environment variable.
5. Falls back to `default`.

When `CAIRNIQ_AUTH_REQUIRED` is on, anything outside the public allow-list needs a valid token: browser page requests are redirected to `/login?next=…`, API calls get a JSON `401`.

Per-request state is stored in a `ContextVar` and reset on response so background tasks see the same profile.

---

## Page Routes (HTML)

These render Jinja2 templates rather than returning JSON:

| Path | Template |
| :--- | :--- |
| `/` | `index.html` (chat console) |
| `/dashboard` | `dashboard.html` (alt dashboard layout) |
| `/journal` | `trade_journal.html` |
| `/context` | `context_and_graph.html` (three tabs: profile, learned, graph) |
| `/portfolio` | `portfolio_editor.html` |
| `/settings` | `terminal_settings.html` |
| `/alerts` | `alerts.html` (alerts inbox, with its own live socket) |
| `/recommendations` | `recommendations.html` (Advisor Ledger scorecard) |
| `/review` | `weekly_review.html` (the weekly one-page review) |
| `/login` | `login.html` |

---

## WebSocket

`/ws` carries server-pushed events to any open page: run status ticks, the live activity/reasoning trace, and new alerts (which drive the nav badge without a poll). Because HTTP middleware does not run for WebSockets, the token is verified on the socket handshake itself when auth is required.

---

## Frontend → Backend Logging Bridge

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/logs/frontend` | Browser-side errors and structured events get forwarded into the same JSONL pipeline used by server-side logs (`logs/frontend/frontend.jsonl`). |

---

## Security Notes

- All mutating endpoints (`POST` / `DELETE` / `PUT` / `PATCH`) are CSRF-protected via origin checking against `ALLOWED_ORIGINS`. Requests from untrusted origins receive `403`.
- In demo mode (`DEMO_MODE=true`), `/api/settings/save` and `/api/profile/switch` return `403` to prevent test users from clobbering real configuration.
- The `/static/` mount and HTML pages are publicly served — only authenticate the JSON API surface if you expose CairnIQ on a network.
