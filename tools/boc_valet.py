"""
Bank of Canada Valet API (Advisor Roadmap Theme 5.7).

Free, keyless, straight from the institution that sets the rate. This replaces
FRED's OECD re-publications of Canadian data (IRSTCI01CAM156N / CPALTT01CAM661S),
which are monthly, lag the source by weeks, and carry none of the series the BoC
actually steers by. Four signals:

  1. **Policy rate** (V39079, the target for the overnight rate) with the date and
     size of the last change — a step function, so "when did it last move" is the
     signal, not the level alone.
  2. **CORRA** (AVG.INTWO) — Canada's realized overnight funding rate. Its spread
     to the target is the CAD money-market stress gauge; FRED has no equivalent.
  3. **Core inflation** — CPI-trim and CPI-median, the two measures the BoC itself
     names in its rate decisions, already published as YoY %. Headline CPI is an
     index level here, so its YoY is computed against the same month a year prior.
  4. **Bank lending rates** — prime, conventional 1/3/5-year mortgages, and 1/3/5-year
     GICs. The GIC leg gives `construct_bond_ladder` a real CAD curve instead of a
     flat policy-rate proxy.

Plus the BoC-vs-Fed divergence, which is the single most decision-relevant number
for a CAD-based investor holding USD assets.

Three disciplines, each from a prior incident in this codebase:

  * **`requests`, never raw `urllib`.** On the deployment Mac's framework Python
    `ssl.get_default_verify_paths().cafile` is None, so every raw-urllib HTTPS call
    dies with CERTIFICATE_VERIFY_FAILED. `requests` bundles certifi and works. This
    already silently killed one screener integration (fixed 172bfba).

  * **Freshness is stamped at FETCH, and staleness is a refusal.** The network
    payload is stamped via `tools.freshness.stamp` *before* it is cached, so a cache
    hit replays the original fetch time rather than the read time. Separately, every
    series carries its own publication cadence: an observation older than that
    cadence allows is returned as `unavailable(...)` with the value moved to
    `last_value`, so a caller reading `["value"]` gets nothing rather than a stale
    number wearing a current label.

  * **Unavailable stays unavailable.** API down, series missing, value non-numeric,
    or too old → `tools.tool_errors.unavailable`. Never a fabricated constant, and
    never a fallback dressed as live data. Derived signals (BoC-vs-Fed divergence)
    refuse to compute when either input is unavailable rather than half-computing.

One shape note verified against the live API: when series of different frequencies
are requested together, Valet does NOT return one globally date-sorted list — a
daily series' rows can appear *after* a weekly series' older rows. Anything that
takes `observations[0]` as "the latest" is wrong. `_parse_observations` therefore
sorts per series.
"""
from datetime import date, datetime
from typing import Any

import requests

import tools.daily_cache as daily_cache
from tools.exception_logger import log_exceptions
from tools.freshness import AS_OF_KEY, stamp
from tools.tool_errors import is_unavailable, unavailable

BASE_URL = "https://www.bankofcanada.ca/valet"
SOURCE = "Bank of Canada Valet"
_TIMEOUT = 15

# Cache TTLs. The Valet publishes on a schedule, so these are deliberately generous:
# re-fetching a daily series every 6h is already twice as often as it can change.
_TTL_DAILY = 6 * 3600
_TTL_WEEKLY = 24 * 3600
_TTL_MONTHLY = 24 * 3600

# ---------------------------------------------------------------------------
# Series catalogue
# ---------------------------------------------------------------------------
# `max_age_days` is the staleness gate, sized to the series' own publication
# cadence plus its lag — NOT to a uniform "data should be fresh" instinct:
#   - business-daily rates: a long weekend plus a stat holiday is 5 days.
#   - weekly bank rates post on Wednesdays; two missed weeks means something broke.
#   - CPI is dated the FIRST of its reference month and released ~3 weeks after that
#     month ends. June CPI (dated 2026-06-01) lands ~July 21 already 50 days "old",
#     and is the newest print available until late August. A 95-day window is the
#     correct one here; anything tighter would reject the current CPI as stale.
SERIES: dict[str, dict[str, Any]] = {
    "policy_rate":     {"id": "V39079",      "label": "Target for the overnight rate", "frequency": "business-daily", "max_age_days": 7},
    "corra":           {"id": "AVG.INTWO",   "label": "CORRA (Canadian Overnight Repo Rate Average)", "frequency": "business-daily", "max_age_days": 7},
    "cpi_trim":        {"id": "CPI_TRIM",    "label": "CPI-trim (YoY %)",   "frequency": "monthly", "max_age_days": 95},
    "cpi_median":      {"id": "CPI_MEDIAN",  "label": "CPI-median (YoY %)", "frequency": "monthly", "max_age_days": 95},
    "cpi_common":      {"id": "CPI_COMMON",  "label": "CPI-common (YoY %)", "frequency": "monthly", "max_age_days": 95},
    "cpi_total_index": {"id": "V41690973",   "label": "Total CPI (index)",  "frequency": "monthly", "max_age_days": 95},
    "prime_rate":      {"id": "V80691311",   "label": "Chartered bank prime rate",  "frequency": "weekly", "max_age_days": 21},
    "mortgage_1y":     {"id": "V80691333",   "label": "Conventional mortgage: 1-year", "frequency": "weekly", "max_age_days": 21},
    "mortgage_3y":     {"id": "V80691334",   "label": "Conventional mortgage: 3-year", "frequency": "weekly", "max_age_days": 21},
    "mortgage_5y":     {"id": "V80691335",   "label": "Conventional mortgage: 5-year", "frequency": "weekly", "max_age_days": 21},
    "gic_1y":          {"id": "V80691339",   "label": "GIC: 1-year", "frequency": "weekly", "max_age_days": 21},
    "gic_3y":          {"id": "V80691340",   "label": "GIC: 3-year", "frequency": "weekly", "max_age_days": 21},
    "gic_5y":          {"id": "V80691341",   "label": "GIC: 5-year", "frequency": "weekly", "max_age_days": 21},
}

_BY_ID = {spec["id"]: {"key": key, **spec} for key, spec in SERIES.items()}

# BoC's inflation-control target: 2% midpoint of a 1-3% band.
INFLATION_TARGET = 2.0
INFLATION_BAND = (1.0, 3.0)


def _today() -> date:
    """Today's calendar date. A module-level seam so tests can pin 'now' without
    freezing the clock globally."""
    return datetime.now().date()


# ---------------------------------------------------------------------------
# HTTP + parsing
# ---------------------------------------------------------------------------
def _valet_get(path: str, params: dict | None = None) -> tuple[Any, str | None]:
    """GET a Valet endpoint. Returns ``(json, None)`` or ``(None, reason)``.

    Uses `requests` on purpose — see the module docstring. Never raises: every
    failure mode becomes a reason string the caller turns into `unavailable`.
    """
    url = f"{BASE_URL}/{path.lstrip('/')}"
    try:
        response = requests.get(url, params=params or {}, timeout=_TIMEOUT)
    except requests.RequestException as e:
        return None, f"Bank of Canada Valet unreachable: {type(e).__name__}"

    if response.status_code != 200:
        # Valet returns a JSON body with a `message` on 4xx (e.g. an unknown series).
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("message"):
                detail = f" — {body['message']}"
        except ValueError:
            pass
        return None, f"Bank of Canada Valet returned HTTP {response.status_code}{detail}"

    try:
        return response.json(), None
    except ValueError:
        return None, "Bank of Canada Valet returned a non-JSON body"


def _parse_observations(payload: Any) -> dict[str, list[dict[str, Any]]]:
    """Valet observations → ``{series_id: [{"date", "value"}, ...]}``, newest first.

    Sorting is per series and done here, never inherited from the response order:
    a mixed-frequency request comes back interleaved, not globally sorted (verified
    live). Rows whose value is absent, blank, or non-numeric are dropped — a series
    that publishes a placeholder must not become a fabricated number downstream.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(payload, dict):
        return out
    for row in payload.get("observations") or []:
        if not isinstance(row, dict):
            continue
        obs_date = row.get("d")
        if not isinstance(obs_date, str) or not obs_date:
            continue
        for series_id, cell in row.items():
            if series_id == "d" or not isinstance(cell, dict):
                continue
            raw = cell.get("v")
            if raw is None or raw == "":
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            out.setdefault(series_id, []).append({"date": obs_date, "value": value})
    for rows in out.values():
        rows.sort(key=lambda r: r["date"], reverse=True)
    return out


def _fetch(keys: list[str], recent: int, ttl: int) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch (or replay from cache) observations for `keys` from SERIES.

    Returns ``({"series": {...}, "labels": {...}, "_as_of": ...}, None)`` or
    ``(None, reason)``.

    The `stamp` goes on BEFORE the payload is cached, so a cache hit carries the
    original fetch time. That is what makes `_as_of` on every derived result an
    honest observation stamp rather than a read-time restamp — the exact bug
    tools/freshness.py exists to prevent. Failures are never cached.
    """
    ids = [SERIES[key]["id"] for key in keys]
    cache_key = f"boc_valet:{','.join(ids)}:{recent}"

    replayed = daily_cache.get_cached(cache_key, ttl_seconds=ttl)
    if isinstance(replayed, dict) and isinstance(replayed.get("series"), dict):
        return replayed, None

    payload, err = _valet_get(f"observations/{','.join(ids)}/json", {"recent": recent})
    if err:
        return None, err

    parsed = _parse_observations(payload)
    if not parsed:
        return None, "Bank of Canada Valet returned no usable observations"

    labels = {}
    detail = payload.get("seriesDetail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        for series_id, meta in detail.items():
            if isinstance(meta, dict):
                # Valet labels some legacy V-code series with the code itself
                # ("V39079"); its `description` is the human name in that case.
                label = str(meta.get("label") or "")
                if not label or label == series_id:
                    label = str(meta.get("description") or label)
                labels[series_id] = label

    result = stamp({"series": parsed, "labels": labels})
    daily_cache.set_cached(cache_key, result)
    return result, None


# ---------------------------------------------------------------------------
# Freshness-gated readers
# ---------------------------------------------------------------------------
def _age_days(obs_date: str, today: date | None = None) -> int | None:
    """Calendar days between an observation date and today. None if unparseable."""
    try:
        return ((today or _today()) - date.fromisoformat(obs_date)).days
    except (TypeError, ValueError):
        return None


def read_series(fetched: dict[str, Any], key: str, today: date | None = None) -> dict[str, Any]:
    """Latest observation for `key`, or `unavailable` when missing or too old.

    A stale series does NOT come back with a `value`. Its number is moved to
    `last_value` and the payload is flagged `unavailable`, because a consumer that
    reads `["value"]` must get nothing rather than a number it will narrate as
    current. That is the same failure mode as the `data_freshness: "Real-time"`
    label on a four-hour-old quote.
    """
    spec = SERIES[key]
    rows = (fetched.get("series") or {}).get(spec["id"]) or []
    if not rows:
        return unavailable(SOURCE, f"{spec['label']} returned no observations", series=spec["id"])

    latest = rows[0]
    age = _age_days(latest["date"], today)
    if age is None:
        return unavailable(
            SOURCE,
            f"{spec['label']} has an unparseable observation date ({latest['date']!r})",
            series=spec["id"],
        )
    if age > spec["max_age_days"]:
        return unavailable(
            SOURCE,
            f"{spec['label']} last published {latest['date']} ({age} days ago), beyond the "
            f"{spec['max_age_days']}-day window for a {spec['frequency']} series — "
            "refusing to present it as current",
            series=spec["id"],
            stale=True,
            last_value=latest["value"],
            last_observation_date=latest["date"],
            observation_age_days=age,
        )

    return {
        "series": spec["id"],
        "label": fetched.get("labels", {}).get(spec["id"]) or spec["label"],
        "value": latest["value"],
        "observation_date": latest["date"],
        "observation_age_days": age,
        "frequency": spec["frequency"],
        "history": rows,
    }


def _value_of(reading: Any) -> float | None:
    """The number, or None if the reading is unavailable. Never raises."""
    if isinstance(reading, dict) and not is_unavailable(reading):
        v = reading.get("value")
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _public(reading: dict[str, Any]) -> dict[str, Any]:
    """A reading without its `history` (kept internal — it bloats LLM context)."""
    return {k: v for k, v in reading.items() if k != "history"}


def _find_year_ago(rows: list[dict[str, Any]], anchor: str) -> dict[str, Any] | None:
    """The observation from the same calendar month, one year before `anchor`.

    Date-matched rather than positional, for the same reason fred_api._find_year_ago
    is: a skipped or placeholder period silently shifts any positional window.
    """
    try:
        year, month = anchor[:7].split("-")
        target = f"{int(year) - 1}-{month}"
    except (ValueError, AttributeError):
        return None
    for row in rows:
        if row["date"][:7] == target:
            return row
    return None


def _one_year_before(day: date) -> date:
    """`day` minus one year, Feb-29-safe (2028-02-29 → 2027-02-28)."""
    try:
        return day.replace(year=day.year - 1)
    except ValueError:
        return day.replace(year=day.year - 1, day=28)


def _nearest_on_or_before(rows: list[dict[str, Any]], cutoff: date) -> dict[str, Any] | None:
    """Newest observation dated on or before `cutoff` (rows are newest-first)."""
    for row in rows:
        try:
            if date.fromisoformat(row["date"]) <= cutoff:
                return row
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# 1. Policy rate + CORRA
# ---------------------------------------------------------------------------
@log_exceptions()
def get_boc_policy_rate() -> dict[str, Any]:
    """
    Bank of Canada policy rate (target for the overnight rate) with the date and
    size of its last move, plus CORRA and CORRA's spread to the target.

    The level alone under-describes a step function: "2.25%, cut 25bp on 2026-06-04"
    is the decision-relevant statement. The CORRA spread is the CAD funding-stress
    gauge — CORRA persistently above target means collateral is tight.
    """
    fetched, err = _fetch(["policy_rate", "corra"], recent=400, ttl=_TTL_DAILY)
    if err:
        return unavailable(SOURCE, err)

    policy = read_series(fetched, "policy_rate")
    if is_unavailable(policy):
        return policy

    rows = policy["history"]
    current = policy["value"]

    # Last change: walk back to the first observation at a different level. The
    # change took effect on the earliest date still carrying the CURRENT level.
    last_change: dict[str, Any] | None = None
    for i, row in enumerate(rows):
        if row["value"] != current:
            previous = row["value"]
            effective = rows[i - 1]["date"]
            bps = round((current - previous) * 100)
            last_change = {
                "date": effective,
                "from": previous,
                "to": current,
                "change_bps": bps,
                "direction": "cut" if bps < 0 else "hike",
            }
            break

    year_ago = _nearest_on_or_before(rows, _one_year_before(_today()))
    change_1y_bps = round((current - year_ago["value"]) * 100) if year_ago else None

    result: dict[str, Any] = {
        "indicator": "Bank of Canada policy rate",
        "source": SOURCE,
        "policy_rate": current,
        "policy_rate_pct": f"{current:.2f}%",
        "observation_date": policy["observation_date"],
        "observation_age_days": policy["observation_age_days"],
        "last_change": last_change or "No change within the observation window",
        "change_1y_bps": change_1y_bps,
        AS_OF_KEY: fetched.get(AS_OF_KEY),
    }

    corra = read_series(fetched, "corra")
    corra_value = _value_of(corra)
    if corra_value is None:
        result["corra"] = corra
    else:
        spread_bps = round((corra_value - current) * 100, 1)
        result["corra"] = _public(corra)
        result["corra_vs_target_bps"] = spread_bps
        result["funding_conditions"] = (
            "Tight — CORRA is trading materially above the target, which signals collateral scarcity"
            if spread_bps >= 5
            else "Soft — CORRA is trading materially below the target, which signals excess settlement balances"
            if spread_bps <= -5
            else "Normal — CORRA is tracking the target"
        )

    result["interpretation"] = (
        "The target for the overnight rate is the BoC's policy instrument; it moves only on "
        "scheduled decision dates. CORRA is the realized overnight funding rate — its spread "
        "to the target, not its level, is the money-market stress signal."
    )
    return result


# ---------------------------------------------------------------------------
# 2. Inflation
# ---------------------------------------------------------------------------
@log_exceptions()
def get_boc_inflation() -> dict[str, Any]:
    """
    Canadian inflation as the Bank of Canada measures it: CPI-trim and CPI-median
    (the two core measures it cites in rate decisions) plus headline CPI YoY.

    CPI-trim/median/common arrive already expressed as YoY %. Headline arrives as an
    index level, so its YoY is computed against the same calendar month a year prior.
    """
    fetched, err = _fetch(
        ["cpi_trim", "cpi_median", "cpi_common", "cpi_total_index"], recent=18, ttl=_TTL_MONTHLY
    )
    if err:
        return unavailable(SOURCE, err)

    trim = read_series(fetched, "cpi_trim")
    median = read_series(fetched, "cpi_median")
    common = read_series(fetched, "cpi_common")
    total = read_series(fetched, "cpi_total_index")

    trim_v, median_v = _value_of(trim), _value_of(median)
    if trim_v is None and median_v is None:
        # Both core measures gone means there is nothing here worth reporting;
        # returning a headline-only payload would invite it to be read as "core".
        return unavailable(
            SOURCE,
            "Neither CPI-trim nor CPI-median is currently available: "
            f"{trim.get('reason') if is_unavailable(trim) else ''} "
            f"{median.get('reason') if is_unavailable(median) else ''}".strip(),
        )

    core_pair = [v for v in (trim_v, median_v) if v is not None]
    core_avg = round(sum(core_pair) / len(core_pair), 2)

    result: dict[str, Any] = {
        "indicator": "Canadian inflation (Bank of Canada)",
        "source": SOURCE,
        "cpi_trim": _public(trim) if trim_v is not None else trim,
        "cpi_median": _public(median) if median_v is not None else median,
        "cpi_common": _public(common) if _value_of(common) is not None else common,
        "core_average_pct": core_avg,
        "target_pct": INFLATION_TARGET,
        "control_band_pct": list(INFLATION_BAND),
        "vs_target": (
            "Above the 1-3% control band" if core_avg > INFLATION_BAND[1]
            else "Below the 1-3% control band" if core_avg < INFLATION_BAND[0]
            else "Above the 2% midpoint, inside the band" if core_avg > INFLATION_TARGET
            else "Below the 2% midpoint, inside the band" if core_avg < INFLATION_TARGET
            else "At the 2% target"
        ),
        AS_OF_KEY: fetched.get(AS_OF_KEY),
    }

    # Headline YoY from the index level.
    if _value_of(total) is not None:
        prior = _find_year_ago(total["history"], total["observation_date"])
        if prior and prior["value"]:
            yoy = ((total["value"] - prior["value"]) / prior["value"]) * 100
            result["headline_cpi_yoy_pct"] = round(yoy, 1)
            result["headline_observation_date"] = total["observation_date"]
        else:
            result["headline_cpi_yoy_pct"] = None
            result["headline_note"] = (
                "Headline YoY not computed: no CPI index observation from the same month a year prior."
            )
    else:
        result["headline_cpi_yoy_pct"] = None
        result["headline_note"] = total.get("reason") if is_unavailable(total) else "Headline CPI unavailable."

    result["interpretation"] = (
        "CPI-trim and CPI-median are the BoC's preferred core measures — they are what the "
        "rate statements cite, not headline CPI. Core above 3% argues against cuts; core "
        "below 2% opens room for them."
    )
    return result


# ---------------------------------------------------------------------------
# 3. Bank lending / deposit rates
# ---------------------------------------------------------------------------
@log_exceptions()
def get_boc_bank_rates() -> dict[str, Any]:
    """
    Chartered-bank prime, conventional 1/3/5-year mortgage rates, and 1/3/5-year GIC
    rates, as posted weekly to the Bank of Canada.

    These are posted reference rates, not offers: a broker will beat the posted
    mortgage rate and a promo GIC will beat the posted GIC. They are the right
    anchor for planning and the wrong number to quote as an executable price.
    """
    keys = ["prime_rate", "mortgage_1y", "mortgage_3y", "mortgage_5y", "gic_1y", "gic_3y", "gic_5y"]
    fetched, err = _fetch(keys, recent=16, ttl=_TTL_WEEKLY)
    if err:
        return unavailable(SOURCE, err)

    readings = {key: read_series(fetched, key) for key in keys}
    if all(is_unavailable(r) for r in readings.values()):
        return unavailable(SOURCE, "No chartered-bank rate series returned a current observation")

    result: dict[str, Any] = {
        "indicator": "Canadian chartered-bank rates",
        "source": SOURCE,
        "currency": "CAD",
        AS_OF_KEY: fetched.get(AS_OF_KEY),
    }
    for key, reading in readings.items():
        result[key] = _public(reading)

    gic_curve = {
        tenor: _value_of(readings[f"gic_{tenor}y"])
        for tenor in (1, 3, 5)
    }
    result["gic_curve"] = {t: v for t, v in gic_curve.items() if v is not None}
    result["note"] = (
        "Posted reference rates, not quotes. Brokered mortgage rates typically price below "
        "the posted conventional rate, and promotional GICs above the posted GIC rate."
    )
    return result


@log_exceptions()
def get_cad_gic_curve() -> dict[str, Any]:
    """
    A 1-5 year CAD GIC curve for ladder construction, from the BoC's posted
    chartered-bank GIC series.

    The BoC posts 1, 3 and 5-year points only. The 2 and 4-year rungs are linearly
    interpolated and flagged as such — an interpolation between two real observations
    is a stated approximation, unlike the flat policy-rate proxy this replaces, which
    asserted a shape the data never showed.
    """
    rates = get_boc_bank_rates()
    if is_unavailable(rates):
        return rates

    anchors = rates.get("gic_curve") or {}
    # A JSON round-trip through the daily cache turns int keys into strings.
    anchors = {int(k): float(v) for k, v in anchors.items()}
    if not {1, 3, 5} <= set(anchors):
        return unavailable(
            SOURCE,
            "Incomplete GIC curve: the Bank of Canada 1/3/5-year posted GIC series did not all "
            f"return a current observation (have: {sorted(anchors) or 'none'})",
        )

    curve = {
        1: anchors[1],
        2: round((anchors[1] + anchors[3]) / 2, 3),
        3: anchors[3],
        4: round((anchors[3] + anchors[5]) / 2, 3),
        5: anchors[5],
    }
    observed = rates.get("gic_5y", {}).get("observation_date")
    return {
        "curve": curve,
        "currency": "CAD",
        "observation_date": observed,
        "interpolated_tenors": [2, 4],
        "source": f"{SOURCE} (posted chartered-bank GIC rates)",
        AS_OF_KEY: rates.get(AS_OF_KEY),
    }


# ---------------------------------------------------------------------------
# 4. BoC vs Fed divergence
# ---------------------------------------------------------------------------
def _fed_policy_rate() -> tuple[float | None, str | None, float | None, str | None]:
    """(rate, as_of, change_1y_bps, reason_unavailable) for the US Fed funds rate.

    Reads the existing FRED path so there is one Fed number in the codebase. The
    fallback branches matter: `get_fed_funds_rate` answers with a hardcoded constant
    when FRED_API_KEY is missing, and comparing a live BoC rate against a hardcoded
    US number would manufacture a "divergence" out of a stale literal. Any payload
    carrying an error or a fallback note is therefore treated as unavailable.
    """
    try:
        from tools.fred_api import get_fed_funds_rate
        payload = get_fed_funds_rate()
    except Exception as e:
        return None, None, None, f"US Fed funds rate lookup failed: {type(e).__name__}"

    if not isinstance(payload, dict):
        return None, None, None, "US Fed funds rate lookup returned no data"
    if payload.get("error"):
        return None, None, None, str(payload["error"])
    note = str(payload.get("note") or "")
    if "fallback" in note.lower() or "cached" in note.lower():
        return None, None, None, f"US Fed funds rate is not live ({note.strip()})"

    def _pct(raw: Any) -> float | None:
        if not isinstance(raw, str):
            return None
        try:
            return float(raw.replace("%", "").replace("+", "").strip())
        except ValueError:
            return None

    rate = _pct(payload.get("current_rate"))
    if rate is None:
        return None, None, None, "US Fed funds rate was not a parseable number"
    change_1y = _pct(payload.get("change_1y"))
    return rate, payload.get("as_of"), (round(change_1y * 100) if change_1y is not None else None), None


@log_exceptions()
def get_boc_fed_divergence() -> dict[str, Any]:
    """
    BoC policy rate vs the US Fed funds rate — the spread, the direction each has
    moved over a year, and the CAD funding-differential mechanism that follows.

    Refuses to compute when either leg is unavailable or not live, rather than
    reporting a spread against a placeholder.
    """
    boc = get_boc_policy_rate()
    if is_unavailable(boc):
        return unavailable(SOURCE, f"Canadian leg unavailable: {boc.get('reason')}")

    fed_rate, fed_as_of, fed_change_1y_bps, fed_err = _fed_policy_rate()
    if fed_err:
        return unavailable(
            SOURCE,
            f"US leg unavailable, so no divergence can be stated: {fed_err}",
            boc_policy_rate=boc.get("policy_rate"),
            boc_observation_date=boc.get("observation_date"),
        )

    boc_rate = boc["policy_rate"]
    spread_bps = round((boc_rate - fed_rate) * 100)
    boc_change_1y_bps = boc.get("change_1y_bps")

    if abs(spread_bps) < 25:
        stance = "Aligned — the two policy rates are within 25bp of each other"
    elif spread_bps < 0:
        stance = f"BoC is {abs(spread_bps)}bp BELOW the Fed"
    else:
        stance = f"BoC is {spread_bps}bp ABOVE the Fed"

    relative = None
    if boc_change_1y_bps is not None and fed_change_1y_bps is not None:
        delta = boc_change_1y_bps - fed_change_1y_bps
        if abs(delta) < 25:
            relative = "Moving together — both policy rates have shifted by a similar amount over the past year"
        elif delta < 0:
            relative = f"BoC has eased {abs(delta)}bp more than the Fed over the past year"
        else:
            relative = f"BoC has tightened {delta}bp more than the Fed over the past year"

    if spread_bps <= -50:
        cad_mechanism = (
            "A negative rate differential favours holding USD over CAD on carry, which is "
            "structural downward pressure on the CAD. For a CAD-base portfolio that raises the "
            "translated value of unhedged USD holdings and raises the cost of buying more of them."
        )
    elif spread_bps >= 50:
        cad_mechanism = (
            "A positive rate differential favours holding CAD over USD on carry, which supports "
            "the CAD. For a CAD-base portfolio that lowers the translated value of unhedged USD "
            "holdings and makes buying more of them cheaper."
        )
    else:
        cad_mechanism = (
            "The rate differential is small enough that carry is not currently the dominant "
            "driver of USD/CAD; terms of trade and risk sentiment are likely to matter more."
        )

    return {
        "indicator": "BoC vs Fed policy divergence",
        "boc_policy_rate": boc_rate,
        "boc_source": SOURCE,
        "boc_observation_date": boc.get("observation_date"),
        "fed_funds_rate": fed_rate,
        "fed_source": "FRED (FEDFUNDS, effective rate)",
        "fed_observation_date": fed_as_of,
        "spread_bps": spread_bps,
        "stance": stance,
        "boc_change_1y_bps": boc_change_1y_bps,
        "fed_change_1y_bps": fed_change_1y_bps,
        "relative_direction_1y": relative,
        "cad_mechanism": cad_mechanism,
        "comparison_note": (
            "The BoC figure is a policy TARGET; the Fed figure is the EFFECTIVE funds rate and a "
            "monthly average, so it lags an intra-month FOMC move. Treat sub-25bp spreads as noise."
        ),
        AS_OF_KEY: boc.get(AS_OF_KEY),
    }


# ---------------------------------------------------------------------------
# 5. Composite
# ---------------------------------------------------------------------------
@log_exceptions()
def get_canada_macro_snapshot() -> dict[str, Any]:
    """
    One-call Canadian macro picture from the Bank of Canada: policy rate + CORRA,
    core inflation, chartered-bank rates, and BoC-vs-Fed divergence.

    Each block stands alone — one unavailable block does not sink the others, and an
    unavailable block is reported as unavailable rather than omitted, so a reader can
    tell "not published" from "not checked".
    """
    def _block(name: str, fn):
        # One block raising must not take the other three with it. Every other
        # entry point lets an unexpected exception propagate on purpose (so an
        # unstubbed test fails loudly rather than quietly returning "unavailable");
        # the composite is the one place where partial degradation is correct.
        try:
            return fn()
        except Exception as e:
            return unavailable(SOURCE, f"{name} lookup raised {type(e).__name__}: {e}")

    snapshot: dict[str, Any] = {
        "region": "Canada",
        "currency": "CAD",
        "source": SOURCE,
        "policy": _block("Policy rate", get_boc_policy_rate),
        "inflation": _block("Inflation", get_boc_inflation),
        "bank_rates": _block("Bank rates", get_boc_bank_rates),
    }
    snapshot["vs_fed"] = _block("BoC-vs-Fed divergence", get_boc_fed_divergence)

    unavailable_blocks = [name for name in ("policy", "inflation", "bank_rates", "vs_fed")
                          if is_unavailable(snapshot[name])]
    if unavailable_blocks:
        snapshot["unavailable_blocks"] = unavailable_blocks
    if len(unavailable_blocks) == 4:
        return unavailable(SOURCE, "No Bank of Canada series could be retrieved")

    headline = []
    policy = snapshot["policy"]
    if not is_unavailable(policy):
        headline.append(f"BoC policy rate {policy['policy_rate_pct']} (as of {policy['observation_date']})")
    inflation = snapshot["inflation"]
    if not is_unavailable(inflation):
        headline.append(f"core inflation {inflation['core_average_pct']}% ({inflation['vs_target'].lower()})")
    vs_fed = snapshot["vs_fed"]
    if not is_unavailable(vs_fed):
        headline.append(vs_fed["stance"])
    snapshot["summary"] = "; ".join(headline) if headline else "No current Bank of Canada readings."
    snapshot[AS_OF_KEY] = policy.get(AS_OF_KEY) if isinstance(policy, dict) else None
    return snapshot


if __name__ == "__main__":  # pragma: no cover - manual verification helper
    import json
    print(json.dumps(get_canada_macro_snapshot(), indent=2, default=str))
