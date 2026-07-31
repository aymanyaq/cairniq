# Opportunity Scanner Configuration (`funnel_config.json`)

The broad-market opportunity scanner (Opportunity Funnel V2) externalizes all of
its tuning knobs into a single JSON file. **You never have to touch the code to
re-tune the funnel** — change the file, restart, done.

This is a reference for that file. The pipeline design spec is no longer
distributed with the repo; `tools/opportunity_scanner.py` is the source of truth
for stage behavior.

---

## TL;DR

- **Location:** `user_data/funnel_config.json`
- **Template:** [`funnel_config.example.json`](../../funnel_config.example.json) (checked into the repo root)
- **It's optional.** If the file is missing, the scanner runs on safe built-in
  defaults and prints one warning. Nothing breaks.
- **It's created for you.** The installer (`install.ps1`) and the first server
  startup both seed `user_data/funnel_config.json` from the example if it doesn't
  exist yet. So normally you'll already have a copy to edit.
- **Partial files are fine.** Anything you omit falls back to the default for that
  key (deep merge), so a config containing only `{"final_top_n": 10}` is valid.

---

## How it loads

On every broad scan the scanner calls `_load_funnel_config()`
(`tools/opportunity_scanner.py`):

1. Read `user_data/funnel_config.json`.
2. **Deep-merge** it over the built-in `_DEFAULT_FUNNEL_CONFIG` — your values win,
   missing keys keep their defaults (nested objects merge key-by-key, not
   wholesale-replace).
3. If the file is missing → built-in defaults + one loud warning per process.
4. If the file is malformed (bad JSON, non-object root) → built-in defaults + a
   warning naming the error. A broken config can never crash a scan.

Edits take effect on the **next scan** (the file is re-read each run), so you can
iterate without restarting — though restarting is the clean way to confirm.

---

## Creating / editing the file

Normally it's already there (seeded on install/first-run). To create it manually:

```bash
cp funnel_config.example.json user_data/funnel_config.json
# then edit user_data/funnel_config.json
```

To reset to defaults, just delete `user_data/funnel_config.json` — the next
startup re-seeds it from the example.

---

## Field reference

### `theme_weights` — how sector themes are ranked (M2)
Weights for the components that score each sector/theme. They should sum to ~1.0.

| Key | Default | Meaning |
|---|---|---|
| `momentum` | 0.45 | Sector rotation / relative momentum strength |
| `macro` | 0.25 | Macro/liquidity alignment (M2 environment) |
| `catalyst` | 0.20 | Active geo/policy event catalysts hitting the sector |
| `breadth` | 0.10 | Participation/breadth of the move |

*Raise `catalyst` to lean into event-driven rotation; raise `momentum` for a
trend-following tilt.*

### `pillars` — the additive base score (M3)
Maximum points each scoring pillar can contribute. **They sum to 100** = the base
score before flow bonus, entry multiplier, and risk adjustment.

| Key | Default | Meaning |
|---|---|---|
| `theme` | 30 | Strength/cycle-stage of the name's theme |
| `relstr` | 25 | Relative strength (alpha vs SPY, 3-month) |
| `forward` | 25 | Forward fundamentals (valuation, growth, analyst upside) |
| `quality` | 20 | Balance-sheet / margin / cash-flow quality |

*Quality is intentionally the smallest weight — the funnel is accumulation- and
theme-first, not a quality screen. Bump `quality` if you want a more conservative
book.*

### `flow_bonus` — institutional-flow confirmation (capped)
Points added when dark-pool / whale / insider flow **confirms** a name. Capped and
confirmation-only by design (the flow proxy is unvalidated; see spec §5.4).

| Key | Default | Meaning |
|---|---|---|
| `two_plus` | 10 | 2 or more independent flow confirmations |
| `one` | 5 | Exactly 1 flow confirmation |

### `entry_multipliers` — don't-chase gate (M3)
The base+flow score is **multiplied** by the multiplier for the name's entry stage.
This is the funnel's core accumulation-first discipline.

| Key | Default | Meaning |
|---|---|---|
| `accumulation_base` | 1.10 | Basing / pullback near support — *rewarded* |
| `early_breakout` | 1.00 | Clean early breakout — neutral |
| `mid_trend` | 0.85 | Extended into a trend — mild haircut |
| `extended` | 0.40 | Overbought/parabolic (RSI≥75, "EXTENDED" setup, or far over 50DMA) — heavily demoted |

*This is why a name with great flow but RSI 81 (e.g. an already-ripping mega-cap)
gets cut to 40% of its score and usually won't surface — by design.*

### `tiers` — conviction labels
Score thresholds that map a final score to a conviction label.

| Key | Default | Label at/above |
|---|---|---|
| `exceptional` | 100 | "Exceptional" |
| `high` | 80 | "High Conviction" |
| `qualified` | 60 | "Qualified" |
| `watchlist` | 40 | "Watchlist" (below → "Low Interest") |

*Note: 3+ risk flags caps an otherwise-Exceptional/High name at "Qualified
(Risk-Capped)" regardless of these thresholds.*

### `risk_cap` & `concentration` — surfaced risk (M4)
Risk **informs sizing/ranking; it never silently deletes an idea.**

| Key | Default | Meaning |
|---|---|---|
| `risk_cap` | 15 | Max total points the risk overlay can subtract from a score |
| `concentration.threshold_pct` | 25 | Portfolio sector exposure %, above which a concentration penalty starts |
| `concentration.penalty_per_excess_pct` | 1 | Penalty points per percentage-point of exposure above the threshold (then capped by `risk_cap`) |

### Pipeline sizing & universe
| Key | Default | Meaning |
|---|---|---|
| `top_k_themes` | 4 | How many top-ranked themes drive candidate selection |
| `fast_screen_top_n` | 60 | Candidates that pass the cheap technical pre-screen into deep analysis |
| `final_top_n` | 15 | Picks returned to the user from a broad scan |
| `universe_cap` | 250 | Hard cap on the candidate pool size |
| `download_chunk` | 75 | Tickers per batched price download (latency vs. rate-limit balance) |
| `early_mover_max_5d_gain_pct` | 20 | Intraday gainers above this 5-day gain are excluded at universe-seed (avoids chasing parabolas) |
| `extended_over_sma50_pct` | 35 | % above the 50-day MA at which a name is classed `extended` (alongside RSI≥75 / "EXTENDED" setup) |

### `catalyst` — news→catalyst escalation
| Key | Default | Meaning |
|---|---|---|
| `auto_escalation_enabled` | `true` | Whether high-ranking catalysts pre-compute a full event→exposure→scenario report |
| `max_auto_escalations` | 3 | Cap on escalations per cycle |
| `auto_scan_after_news` | `true` | Run the catalyst pass after a news refresh |
| `auto_scan_min_interval_hours` | 6 | Minimum gap between automatic catalyst passes |
| `stale_event_days` | 3 | Age past which a catalyst is no longer treated as actionable |

### `scheduler` — background task gates

Per-task on/off switches. A missing file, block, or key falls back to **enabled**, so this block only needs the tasks you want to turn *off*. Note this is separate from the `SCHEDULER_ENABLED` setting in Settings, which is the per-profile master switch — with that off, none of these run regardless.

| Key | Default | Task |
|---|---|---|
| `exchange_rate` | `true` | Hourly FX refresh |
| `portfolio_snapshot` | `true` | After-close portfolio history point |
| `score_recommendations` | `true` | Daily Advisor Ledger scoring |
| `cache_cleanup` | `true` | Daily cache expiry |
| `premarket_pulse` | `true` | Pre-market market pulse |
| `priority_precompute` | `true` | Today's Priority brief, 07:00–09:25 (the most expensive job — one full reasoning-graph pass per profile) |
| `funnel_signal_scan` | `true` | One portfolio-neutral broad scan after close, feeding the walk-forward signal log |
| `edgar_events` | `true` | Daily 8-K severity + Form 4 cluster-buy poll for held names |
| `intraday_sentinel` | `true` | Zero-LLM market-state change detection |

`scheduler.priority_precompute_profiles` optionally takes a JSON list of profile names to restrict the precompute to. Omit it (the default) to run for every profile. It exists so real profile names live in machine-local, untracked config rather than in source.

### `edgar` — filings pipeline

| Key | Default | Meaning |
|---|---|---|
| `managers_13f` | curated set | `{"Manager Name": CIK}` map overriding the built-in long-horizon managers whose 13F filings feed the institutional universe and `get_institutional_moves` |

---

## Common tuning recipes

**Get more picks per scan**
```json
{ "fast_screen_top_n": 80, "final_top_n": 25 }
```
Widening `fast_screen_top_n` also reduces the chance a quality accumulation setup
is culled before deep analysis (see the `fast-screen-near-miss` log to check
whether that's happening).

**More conservative / quality-tilted book**
```json
{ "pillars": { "theme": 25, "relstr": 20, "forward": 25, "quality": 30 } }
```

**Lean harder into accumulation, punish chasing more**
```json
{ "entry_multipliers": { "accumulation_base": 1.20, "mid_trend": 0.70, "extended": 0.25 } }
```

**Event-driven tilt**
```json
{ "theme_weights": { "momentum": 0.35, "macro": 0.20, "catalyst": 0.35, "breadth": 0.10 } }
```

**Tighter risk discipline**
```json
{ "risk_cap": 25, "concentration": { "threshold_pct": 20, "penalty_per_excess_pct": 2 } }
```

---

## Notes & gotchas

- **Partial overrides only change what you list.** Because of the deep merge, you
  don't need to copy the whole file — keep your `funnel_config.json` to just the
  keys you've changed for a clean diff against defaults.
- **`pillars` should sum to 100** to keep scores on the documented 0–100 scale; the
  code won't stop you from using other sums, but conviction `tiers` assume ~100.
- **Bad JSON is safe but silent-ish** — you'll get a one-line warning and defaults,
  not a crash. If your tweaks "aren't taking," check the startup log for a
  `Funnel config invalid` warning.
- **The flow bonus is deliberately small and capped** — it's an unvalidated proxy.
  Don't crank it expecting it to dominate; it's a tiebreaker, not a thesis.
