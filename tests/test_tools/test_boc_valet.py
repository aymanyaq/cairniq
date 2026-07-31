"""Bank of Canada Valet tests (Advisor Roadmap 5.7) — fully offline.

Only the HTTP boundary is replaced. Every test drives the real production
functions: the real parser, the real freshness gate, the real daily-cache layer
(already per-test isolated by the `isolated_test_profile` fixture), and the real
`construct_bond_ladder` / tool-registry entry points.

Fixtures are structurally faithful to responses verified against the live API on
2026-07-24, including the two shapes that are easy to get wrong:
  * a mixed-frequency request comes back NOT globally date-sorted, and
  * observations carry an optional `r` revision flag beside `v`.
"""
from datetime import date
from unittest.mock import patch

import pytest

import tools.boc_valet as boc
from tools.freshness import AS_OF_KEY, as_of
from tools.tool_errors import is_unavailable

TODAY = date(2026, 7, 24)


@pytest.fixture(autouse=True)
def _pinned_today(monkeypatch):
    """Pin the clock the freshness gate reads, so 'stale' is deterministic."""
    monkeypatch.setattr(boc, "_today", lambda: TODAY)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _valet_payload(series: dict[str, list[tuple[str, object]]], labels: dict | None = None) -> dict:
    """Build a Valet observations payload from {series_id: [(date, value), ...]}.

    Rows are emitted grouped by series, deliberately NOT merged into one
    date-sorted list — that is exactly how the live API answers a mixed-frequency
    request, and anything that trusts the response order will fail here.
    """
    observations = []
    for series_id, rows in series.items():
        for obs_date, value in rows:
            cell = {} if value is None else {"v": value}
            observations.append({"d": obs_date, series_id: cell})
    detail = {sid: {"label": (labels or {}).get(sid, sid), "description": f"desc {sid}"}
              for sid in series}
    return {"observations": observations, "seriesDetail": detail,
            "terms": {"url": "https://www.bankofcanada.ca/terms/"}}


def _flat(series_id: str, dates: list[str], value: object) -> dict[str, list[tuple[str, object]]]:
    return {series_id: [(d, value) for d in dates]}


def _stub_valet(payload, err=None):
    """Patch the single HTTP boundary. Everything above it stays real."""
    return patch.object(boc, "_valet_get", return_value=(payload, err))


# Business-day-ish date ladders ending 2026-07-23 (one day before pinned TODAY).
_DAILY_DATES = ["2026-07-23", "2026-07-22", "2026-07-21", "2026-07-20", "2026-07-17"]


def _policy_payload(current=2.25, previous=2.50, change_index=3, year_ago_value=3.00):
    """Policy-rate step function + CORRA, in the real two-series shape."""
    dates = _DAILY_DATES + ["2026-06-05", "2026-06-04", "2026-06-03", "2025-07-23", "2025-07-22"]
    rows = []
    for i, d in enumerate(dates):
        if d.startswith("2025"):
            rows.append((d, f"{year_ago_value:.2f}"))
        elif i < change_index:
            rows.append((d, f"{current:.2f}"))
        else:
            rows.append((d, f"{previous:.2f}"))
    # Force the change boundary onto a known date: everything from _DAILY_DATES[
    # change_index] backwards (until 2025) sits at `previous`.
    corra = [(d, f"{current + 0.02:.4f}") for d in _DAILY_DATES]
    return _valet_payload({"V39079": rows, "AVG.INTWO": corra},
                          labels={"V39079": "V39079", "AVG.INTWO": "CORRA"})


def _cpi_payload(trim="1.8", median="1.9", common="2.6", index_now="169.0", index_prior="165.0"):
    months = ["2026-06-01", "2026-05-01", "2026-04-01"]
    series = {}
    if trim is not None:
        series["CPI_TRIM"] = [(m, trim) for m in months]
    if median is not None:
        series["CPI_MEDIAN"] = [(m, median) for m in months]
    if common is not None:
        series["CPI_COMMON"] = [(m, common) for m in months]
    idx = [("2026-06-01", index_now), ("2026-05-01", "169.6")]
    if index_prior is not None:
        idx.append(("2025-06-01", index_prior))
    series["V41690973"] = idx
    return _valet_payload(series, labels={"CPI_TRIM": "CPI-trim", "CPI_MEDIAN": "CPI-median",
                                          "CPI_COMMON": "CPI-common", "V41690973": "Total CPI"})


_WEEKLY_DATES = ["2026-07-22", "2026-07-15", "2026-07-08"]


def _bank_rates_payload(gic_1="3.10", gic_3="3.40", gic_5="3.80"):
    series = {}
    for sid, val in (("V80691311", "4.45"), ("V80691333", "5.49"), ("V80691334", "6.05"),
                     ("V80691335", "6.09"), ("V80691339", gic_1), ("V80691340", gic_3),
                     ("V80691341", gic_5)):
        if val is not None:
            series[sid] = [(d, val) for d in _WEEKLY_DATES]
    return _valet_payload(series)


# ---------------------------------------------------------------------------
# Transport: requests, not urllib (the CERTIFICATE_VERIFY_FAILED lesson)
# ---------------------------------------------------------------------------
def test_transport_is_requests_not_urllib():
    """The deployment Mac's framework Python has no CA bundle on the raw urllib
    path (ssl.get_default_verify_paths().cafile is None), so any urllib HTTPS call
    dies with CERTIFICATE_VERIFY_FAILED. requests bundles certifi. This already
    broke one integration; pin the choice."""
    import requests as real_requests

    assert boc.requests is real_requests

    with open(boc.__file__) as f:
        text = f.read()
    assert "urllib.request" not in text
    assert "urlopen" not in text


def test_valet_get_calls_requests_with_timeout():
    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"observations": []}

    with patch.object(boc.requests, "get", return_value=Resp()) as mock_get:
        payload, err = boc._valet_get("observations/V39079/json", {"recent": 2})

    assert err is None and payload == {"observations": []}
    (url,), kwargs = mock_get.call_args
    assert url == "https://www.bankofcanada.ca/valet/observations/V39079/json"
    assert kwargs["params"] == {"recent": 2}
    assert kwargs["timeout"] == boc._TIMEOUT


def test_valet_get_surfaces_http_error_message():
    class Resp:
        status_code = 404

        @staticmethod
        def json():
            return {"message": "Series NOPE_XYZ not found."}

    with patch.object(boc.requests, "get", return_value=Resp()):
        payload, err = boc._valet_get("observations/NOPE_XYZ/json")

    assert payload is None
    assert "404" in err and "NOPE_XYZ not found" in err


def test_valet_get_transport_failure_is_a_reason_not_an_exception():
    import requests as real_requests

    with patch.object(boc.requests, "get", side_effect=real_requests.ConnectionError("boom")):
        payload, err = boc._valet_get("observations/V39079/json")

    assert payload is None
    assert "unreachable" in err and "ConnectionError" in err


def test_valet_get_non_json_body():
    class Resp:
        status_code = 200

        @staticmethod
        def json():
            raise ValueError("not json")

    with patch.object(boc.requests, "get", return_value=Resp()):
        payload, err = boc._valet_get("observations/V39079/json")

    assert payload is None and "non-JSON" in err


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parse_sorts_per_series_ignoring_response_order():
    """Verified live: asking for a daily and a weekly series together returns the
    daily rows AFTER the weekly ones. observations[0] is not 'the latest'."""
    payload = {"observations": [
        {"d": "2026-07-22", "V80691335": {"v": "6.09"}, "V39079": {"v": "2.25"}},
        {"d": "2026-07-15", "V80691335": {"v": "6.09"}},
        {"d": "2026-07-08", "V80691335": {"v": "6.05"}},
        {"d": "2026-07-23", "V39079": {"v": "2.25"}},   # newer, but last in the list
        {"d": "2026-07-21", "V39079": {"v": "2.50"}},
    ]}
    parsed = boc._parse_observations(payload)

    assert [r["date"] for r in parsed["V39079"]] == ["2026-07-23", "2026-07-22", "2026-07-21"]
    assert [r["date"] for r in parsed["V80691335"]] == ["2026-07-22", "2026-07-15", "2026-07-08"]
    assert parsed["V39079"][0]["value"] == 2.25


def test_parse_drops_unusable_cells_and_keeps_revised_ones():
    payload = {"observations": [
        {"d": "2026-06-01", "CPI_COMMON": {"r": 1, "v": "2.6"}},   # revision flag: still a number
        {"d": "2026-05-01", "CPI_COMMON": {"v": ""}},              # blank placeholder
        {"d": "2026-04-01", "CPI_COMMON": {"v": "n/a"}},           # non-numeric
        {"d": "2026-03-01", "CPI_COMMON": {}},                     # no value at all
        {"d": "", "CPI_COMMON": {"v": "9.9"}},                     # no date
        "not-a-row",
    ]}
    parsed = boc._parse_observations(payload)

    assert parsed == {"CPI_COMMON": [{"date": "2026-06-01", "value": 2.6}]}


def test_parse_tolerates_garbage_payload():
    assert boc._parse_observations(None) == {}
    assert boc._parse_observations({"observations": None}) == {}


# ---------------------------------------------------------------------------
# Freshness: stamped at FETCH, replayed through the cache
# ---------------------------------------------------------------------------
def test_fetch_stamps_before_caching_and_replays_the_original_stamp():
    """The stamp must be written inside the payload that goes INTO the cache, so a
    cache hit reports when the data was fetched rather than when it was read."""
    stored = {}
    real_set = boc.daily_cache.set_cached

    def spy(key, data):
        stored[key] = data
        real_set(key, data)

    with _stub_valet(_policy_payload()) as mock_get, \
            patch.object(boc.daily_cache, "set_cached", side_effect=spy):
        first, err1 = boc._fetch(["policy_rate", "corra"], recent=400, ttl=boc._TTL_DAILY)
        second, err2 = boc._fetch(["policy_rate", "corra"], recent=400, ttl=boc._TTL_DAILY)

    assert err1 is None and err2 is None
    # The cached object itself carries the stamp — not something added on read.
    assert len(stored) == 1
    assert AS_OF_KEY in next(iter(stored.values()))
    # Second call was served from cache and replayed the ORIGINAL fetch time.
    assert mock_get.call_count == 1
    assert as_of(second) is not None
    assert second[AS_OF_KEY] == first[AS_OF_KEY]


def test_fetch_failures_are_never_cached():
    with _stub_valet(None, err="Bank of Canada Valet unreachable: ConnectionError") as bad:
        payload, err = boc._fetch(["policy_rate"], recent=5, ttl=boc._TTL_DAILY)
    assert payload is None and "unreachable" in err
    assert bad.call_count == 1

    with _stub_valet(_valet_payload(_flat("V39079", ["2026-07-23"], "2.25"))) as good:
        payload, err = boc._fetch(["policy_rate"], recent=5, ttl=boc._TTL_DAILY)
    assert err is None and payload["series"]["V39079"][0]["value"] == 2.25
    assert good.call_count == 1  # the earlier failure did not poison the cache


def test_fetch_empty_observations_is_unavailable_not_empty_success():
    with _stub_valet({"observations": []}):
        payload, err = boc._fetch(["policy_rate"], recent=5, ttl=boc._TTL_DAILY)
    assert payload is None and "no usable observations" in err


def test_fetch_uses_description_when_valet_labels_a_series_by_its_own_code():
    payload = _valet_payload(_flat("V39079", ["2026-07-23"], "2.25"))
    payload["seriesDetail"]["V39079"] = {"label": "V39079", "description": "Target for the overnight rate"}
    with _stub_valet(payload):
        fetched, _ = boc._fetch(["policy_rate"], recent=5, ttl=boc._TTL_DAILY)
    assert fetched["labels"]["V39079"] == "Target for the overnight rate"


# ---------------------------------------------------------------------------
# Freshness gate: stale is a refusal, and the window follows the cadence
# ---------------------------------------------------------------------------
def test_stale_series_returns_unavailable_with_no_value_key():
    """A stale reading must not hand back a `value`. A consumer reading ["value"]
    has to come away with nothing rather than a number it will narrate as current
    — the same failure mode as `data_freshness: "Real-time"` on an EOD print."""
    with _stub_valet(_valet_payload(_flat("V39079", ["2026-06-30"], "2.75"))):
        fetched, _ = boc._fetch(["policy_rate"], recent=5, ttl=boc._TTL_DAILY)

    reading = boc.read_series(fetched, "policy_rate")

    assert is_unavailable(reading)
    assert "value" not in reading
    assert reading["last_value"] == 2.75
    assert reading["last_observation_date"] == "2026-06-30"
    assert reading["observation_age_days"] == 24
    assert "7-day window" in reading["reason"] and "business-daily" in reading["reason"]
    assert boc._value_of(reading) is None


def test_monthly_cpi_50_days_old_is_current_not_stale():
    """June CPI is dated 2026-06-01 and is the newest print available in late July.
    A uniform 'must be days old' rule would wrongly reject the current CPI — the
    freshness window has to follow each series' own publication cadence."""
    with _stub_valet(_cpi_payload()):
        fetched, _ = boc._fetch(["cpi_trim"], recent=18, ttl=boc._TTL_MONTHLY)

    reading = boc.read_series(fetched, "cpi_trim")

    assert not is_unavailable(reading)
    assert reading["observation_age_days"] == 53
    assert reading["value"] == 1.8


def test_missing_series_is_unavailable():
    with _stub_valet(_valet_payload(_flat("V39079", ["2026-07-23"], "2.25"))):
        fetched, _ = boc._fetch(["policy_rate"], recent=5, ttl=boc._TTL_DAILY)
    reading = boc.read_series(fetched, "corra")
    assert is_unavailable(reading) and "no observations" in reading["reason"]


def test_unparseable_observation_date_is_unavailable():
    fetched = {"series": {"V39079": [{"date": "not-a-date", "value": 2.25}]}, "labels": {}}
    reading = boc.read_series(fetched, "policy_rate")
    assert is_unavailable(reading) and "unparseable" in reading["reason"]


# ---------------------------------------------------------------------------
# Policy rate + CORRA
# ---------------------------------------------------------------------------
def test_policy_rate_reports_level_last_change_and_corra_spread():
    with _stub_valet(_policy_payload(current=2.25, previous=2.50, change_index=3, year_ago_value=3.00)):
        result = boc.get_boc_policy_rate()

    assert not is_unavailable(result)
    assert result["policy_rate"] == 2.25
    assert result["policy_rate_pct"] == "2.25%"
    assert result["observation_date"] == "2026-07-23"
    # The move took effect on the earliest date still carrying the CURRENT level.
    assert result["last_change"]["date"] == "2026-07-21"
    assert result["last_change"]["from"] == 2.50
    assert result["last_change"]["change_bps"] == -25
    assert result["last_change"]["direction"] == "cut"
    assert result["change_1y_bps"] == -75
    # CORRA is 2bp above target here — inside the normal band.
    assert result["corra"]["value"] == 2.27
    assert result["corra_vs_target_bps"] == 2.0
    assert result["funding_conditions"].startswith("Normal")
    assert result[AS_OF_KEY]


def test_policy_rate_flags_corra_dislocation():
    payload = _policy_payload()
    payload["observations"] = [
        row for row in payload["observations"] if "AVG.INTWO" not in row
    ] + [{"d": d, "AVG.INTWO": {"v": "2.35"}} for d in _DAILY_DATES]

    with _stub_valet(payload):
        result = boc.get_boc_policy_rate()

    assert result["corra_vs_target_bps"] == 10.0
    assert result["funding_conditions"].startswith("Tight")


def test_policy_rate_survives_missing_corra():
    payload = _valet_payload(_flat("V39079", _DAILY_DATES, "2.25"))
    with _stub_valet(payload):
        result = boc.get_boc_policy_rate()

    assert result["policy_rate"] == 2.25
    assert is_unavailable(result["corra"])
    assert "corra_vs_target_bps" not in result
    assert result["last_change"] == "No change within the observation window"


def test_policy_rate_unavailable_when_api_is_down():
    with _stub_valet(None, err="Bank of Canada Valet unreachable: ConnectTimeout"):
        result = boc.get_boc_policy_rate()

    assert is_unavailable(result)
    assert result["source"] == boc.SOURCE
    assert "unreachable" in result["reason"]
    assert "policy_rate" not in result


def test_policy_rate_unavailable_when_stale():
    with _stub_valet(_valet_payload(_flat("V39079", ["2026-05-01"], "2.75"))):
        result = boc.get_boc_policy_rate()

    assert is_unavailable(result) and result.get("stale") is True
    assert "policy_rate" not in result


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------
def test_inflation_reports_core_pair_and_computes_headline_yoy():
    with _stub_valet(_cpi_payload(trim="1.8", median="1.9", index_now="169.0", index_prior="165.0")):
        result = boc.get_boc_inflation()

    assert not is_unavailable(result)
    assert result["cpi_trim"]["value"] == 1.8
    assert result["cpi_median"]["value"] == 1.9
    assert result["core_average_pct"] == 1.85
    assert result["vs_target"] == "Below the 2% midpoint, inside the band"
    # Headline arrives as an index level, so YoY is computed, not read.
    assert result["headline_cpi_yoy_pct"] == 2.4
    assert result["headline_observation_date"] == "2026-06-01"
    # history is internal plumbing and must not leak into LLM context.
    assert "history" not in result["cpi_trim"]


def test_inflation_above_band_is_labelled():
    with _stub_valet(_cpi_payload(trim="3.4", median="3.6")):
        result = boc.get_boc_inflation()
    assert result["core_average_pct"] == 3.5
    assert result["vs_target"] == "Above the 1-3% control band"


def test_inflation_headline_withheld_without_a_year_ago_observation():
    """No year-ago index month means no YoY. Withhold the number and say why —
    never approximate it off the nearest available month."""
    with _stub_valet(_cpi_payload(index_prior=None)):
        result = boc.get_boc_inflation()

    assert result["core_average_pct"] == 1.85
    assert result["headline_cpi_yoy_pct"] is None
    assert "same month a year prior" in result["headline_note"]


def test_inflation_unavailable_when_both_core_measures_are_missing():
    with _stub_valet(_cpi_payload(trim=None, median=None)):
        result = boc.get_boc_inflation()

    assert is_unavailable(result)
    assert "core_average_pct" not in result


def test_inflation_unavailable_when_api_is_down():
    with _stub_valet(None, err="Bank of Canada Valet returned HTTP 503"):
        result = boc.get_boc_inflation()
    assert is_unavailable(result) and "503" in result["reason"]


# ---------------------------------------------------------------------------
# Bank rates + the CAD GIC curve
# ---------------------------------------------------------------------------
def test_bank_rates_reports_posted_rates_and_labels_them_as_reference():
    with _stub_valet(_bank_rates_payload()):
        result = boc.get_boc_bank_rates()

    assert not is_unavailable(result)
    assert result["prime_rate"]["value"] == 4.45
    assert result["mortgage_5y"]["value"] == 6.09
    assert result["gic_curve"] == {1: 3.10, 3: 3.40, 5: 3.80}
    assert "not quotes" in result["note"]
    assert "history" not in result["prime_rate"]


def test_gic_curve_interpolates_the_two_missing_rungs_and_says_so():
    with _stub_valet(_bank_rates_payload(gic_1="3.10", gic_3="3.40", gic_5="3.80")):
        curve = boc.get_cad_gic_curve()

    assert curve["curve"] == {1: 3.10, 2: 3.25, 3: 3.40, 4: 3.60, 5: 3.80}
    assert curve["interpolated_tenors"] == [2, 4]
    assert curve["observation_date"] == "2026-07-22"
    assert "Bank of Canada" in curve["source"]


def test_gic_curve_unavailable_when_an_anchor_tenor_is_missing():
    with _stub_valet(_bank_rates_payload(gic_3=None)):
        curve = boc.get_cad_gic_curve()

    assert is_unavailable(curve)
    assert "Incomplete GIC curve" in curve["reason"]
    assert "curve" not in curve


def test_bank_rates_unavailable_when_every_series_is_stale():
    stale = _valet_payload({sid: [("2026-01-05", "5.00")] for sid in
                            ("V80691311", "V80691333", "V80691334", "V80691335",
                             "V80691339", "V80691340", "V80691341")})
    with _stub_valet(stale):
        result = boc.get_boc_bank_rates()
    assert is_unavailable(result)


# ---------------------------------------------------------------------------
# BoC vs Fed divergence
# ---------------------------------------------------------------------------
_LIVE_FED = {"indicator": "Federal Funds Rate", "current_rate": "3.63%",
             "as_of": "2026-06-01", "change_1y": "-0.75%"}


def test_divergence_reports_spread_direction_and_cad_mechanism():
    with _stub_valet(_policy_payload()), \
            patch("tools.fred_api.get_fed_funds_rate", return_value=_LIVE_FED):
        result = boc.get_boc_fed_divergence()

    assert not is_unavailable(result)
    assert result["boc_policy_rate"] == 2.25
    assert result["fed_funds_rate"] == 3.63
    assert result["spread_bps"] == -138
    assert result["stance"] == "BoC is 138bp BELOW the Fed"
    assert result["boc_change_1y_bps"] == -75
    assert result["fed_change_1y_bps"] == -75
    assert result["relative_direction_1y"].startswith("Moving together")
    assert "favours holding USD over CAD" in result["cad_mechanism"]
    assert "TARGET" in result["comparison_note"]


def test_divergence_detects_relative_easing():
    with _stub_valet(_policy_payload(year_ago_value=4.25)), \
            patch("tools.fred_api.get_fed_funds_rate", return_value=_LIVE_FED):
        result = boc.get_boc_fed_divergence()

    assert result["boc_change_1y_bps"] == -200
    assert result["relative_direction_1y"] == "BoC has eased 125bp more than the Fed over the past year"


def test_divergence_refuses_to_compare_against_a_fred_fallback_constant():
    """get_fed_funds_rate answers with a HARDCODED constant when FRED_API_KEY is
    missing. Differencing a live BoC rate against that literal would manufacture a
    divergence out of a stale number — the exact fabrication class this repo keeps
    getting bitten by. Refuse instead."""
    fallback_fed = dict(_LIVE_FED, note="Using cached fallback data because FRED_API_KEY is not configured.")

    with _stub_valet(_policy_payload()), \
            patch("tools.fred_api.get_fed_funds_rate", return_value=fallback_fed):
        result = boc.get_boc_fed_divergence()

    assert is_unavailable(result)
    assert "not live" in result["reason"]
    assert "spread_bps" not in result and "fed_funds_rate" not in result
    # The Canadian leg it DID prove is still reported, clearly separated.
    assert result["boc_policy_rate"] == 2.25


def test_divergence_unavailable_when_fred_errors():
    with _stub_valet(_policy_payload()), \
            patch("tools.fred_api.get_fed_funds_rate", return_value={"error": "Rate limit on all FRED keys"}):
        result = boc.get_boc_fed_divergence()

    assert is_unavailable(result) and "Rate limit" in result["reason"]


def test_divergence_unavailable_when_the_canadian_leg_is_down():
    with _stub_valet(None, err="Bank of Canada Valet unreachable: ReadTimeout"), \
            patch("tools.fred_api.get_fed_funds_rate", return_value=_LIVE_FED):
        result = boc.get_boc_fed_divergence()

    assert is_unavailable(result) and "Canadian leg unavailable" in result["reason"]


def test_divergence_calls_alignment_when_the_spread_is_noise():
    with _stub_valet(_policy_payload()), \
            patch("tools.fred_api.get_fed_funds_rate", return_value=dict(_LIVE_FED, current_rate="2.35%")):
        result = boc.get_boc_fed_divergence()

    assert result["spread_bps"] == -10
    assert result["stance"].startswith("Aligned")
    assert "not currently the dominant" in result["cad_mechanism"]


# ---------------------------------------------------------------------------
# Composite snapshot
# ---------------------------------------------------------------------------
def _all_series_payload():
    """One payload that answers every _fetch the snapshot makes."""
    merged = {}
    for part in (_policy_payload(), _cpi_payload(), _bank_rates_payload()):
        for row in part["observations"]:
            merged.setdefault("observations", []).append(row)
        merged.setdefault("seriesDetail", {}).update(part["seriesDetail"])
    return merged


def test_snapshot_bundles_every_block_with_a_summary():
    with _stub_valet(_all_series_payload()), \
            patch("tools.fred_api.get_fed_funds_rate", return_value=_LIVE_FED):
        snap = boc.get_canada_macro_snapshot()

    assert not is_unavailable(snap)
    assert snap["currency"] == "CAD"
    assert snap["policy"]["policy_rate"] == 2.25
    assert snap["inflation"]["core_average_pct"] == 1.85
    assert snap["bank_rates"]["gic_curve"][5] == 3.80
    assert snap["vs_fed"]["spread_bps"] == -138
    assert "unavailable_blocks" not in snap
    assert "BoC policy rate 2.25%" in snap["summary"]
    assert "core inflation 1.85%" in snap["summary"]


def test_snapshot_names_its_dead_blocks_rather_than_omitting_them():
    """An omitted block is indistinguishable from a block that was never checked.
    Report unavailability explicitly so silence can't be back-filled."""
    with _stub_valet(_all_series_payload()), \
            patch("tools.fred_api.get_fed_funds_rate", return_value={"error": "FRED down"}):
        snap = boc.get_canada_macro_snapshot()

    assert snap["unavailable_blocks"] == ["vs_fed"]
    assert is_unavailable(snap["vs_fed"])
    assert snap["policy"]["policy_rate"] == 2.25


def test_snapshot_unavailable_when_nothing_can_be_retrieved():
    with _stub_valet(None, err="Bank of Canada Valet unreachable: ConnectionError"), \
            patch("tools.fred_api.get_fed_funds_rate", return_value=_LIVE_FED):
        snap = boc.get_canada_macro_snapshot()

    assert is_unavailable(snap)
    assert "No Bank of Canada series" in snap["reason"]


def test_snapshot_converts_an_unexpected_exception_into_unavailable():
    with patch.object(boc, "get_boc_policy_rate", side_effect=RuntimeError("kaboom")), \
            patch.object(boc, "get_boc_inflation", side_effect=RuntimeError("kaboom")), \
            patch.object(boc, "get_boc_bank_rates", side_effect=RuntimeError("kaboom")), \
            patch.object(boc, "get_boc_fed_divergence", side_effect=RuntimeError("kaboom")):
        snap = boc.get_canada_macro_snapshot()

    assert is_unavailable(snap)


# ---------------------------------------------------------------------------
# Downstream: the CAD bond ladder now rides a real BoC curve
# ---------------------------------------------------------------------------
def test_cad_bond_ladder_uses_the_boc_gic_curve():
    """Drives the real construct_bond_ladder, not the curve helper in isolation."""
    import tools.fixed_income as fixed

    with _stub_valet(_bank_rates_payload(gic_1="3.10", gic_3="3.40", gic_5="3.80")):
        rates, note = fixed._fetch_current_rates("GIC", "CAD")
        ladder = fixed.construct_bond_ladder(100000, "GIC", "CAD")

    assert rates == {1: 3.10, 2: 3.25, 3: 3.40, 4: 3.60, 5: 3.80}
    assert "Bank of Canada" in note and "2026-07-22" in note
    assert "interpolated" in note
    assert ladder["rungs"][0]["rate"] == "3.10%"
    assert ladder["rungs"][4]["rate"] == "3.80%"
    assert ladder["average_yield"] == "3.43%"


def test_cad_bond_ladder_falls_back_to_fred_when_boc_is_down():
    import tools.fixed_income as fixed

    with _stub_valet(None, err="Bank of Canada Valet unreachable: ConnectionError"), \
            patch("tools.fred_api.get_canada_metrics", return_value={"interest_rate": "4.10%"}):
        rates, note = fixed._fetch_current_rates("GIC", "CAD")

    assert rates == {1: 4.10, 2: 4.10, 3: 4.10, 4: 4.10, 5: 4.10}
    assert "FRED" in note and "flat across tenors" in note


# ---------------------------------------------------------------------------
# Downstream: agent surface
# ---------------------------------------------------------------------------
def test_registry_exposes_the_canada_tools_and_they_reach_boc_valet():
    import agent.tool_registry as reg

    names = {t.name for t in reg.MACRO_TOOLS}
    assert {"get_canada_macro", "get_boc_vs_fed"} <= names
    assert {"get_canada_macro", "get_boc_vs_fed"} <= {t.name for t in reg.ALL_TOOLS}

    with _stub_valet(_all_series_payload()), \
            patch("tools.fred_api.get_fed_funds_rate", return_value=_LIVE_FED):
        snapshot = reg.get_canada_macro.invoke({})
        divergence = reg.get_boc_vs_fed.invoke({})

    assert snapshot["policy"]["policy_rate"] == 2.25
    assert divergence["spread_bps"] == -138


def test_tool_relationship_graph_links_the_canada_cluster():
    from agent.tool_retriever import TOOL_RELATIONSHIPS

    assert "get_canada_macro" in TOOL_RELATIONSHIPS["get_macro_overview"]
    assert "get_boc_vs_fed" in TOOL_RELATIONSHIPS["get_canada_macro"]


def test_macro_strategy_prefers_boc_over_freds_oecd_republication():
    """The whole point of 5.7: analyze_macro_context's Canadian block must read the
    Bank of Canada, and must NAME which source it read so the two can never be
    confused for one another."""
    import tools.macro_strategy as strategy

    us_macro = {
        "fed_funds": {"current_rate": "3.63%"},
        "inflation": {"headline_inflation": "2.4%"},
        "unemployment": {"current_rate": "4.3%"},
        "treasury_yields": {"10_year_yield": "4.44%", "2_year_yield": "4.14%"},
        "gdp": {"trend": "Stable"},
    }
    with patch.object(strategy, "get_all_macro_indicators", return_value=us_macro), \
            patch.object(strategy, "get_systemic_risk_indicators",
                         return_value={"crash_risk": "Low", "liquidity_status": "Expanding"}), \
            patch.object(strategy, "get_canada_metrics",
                         return_value={"interest_rate": "9.99%", "inflation": "9.99%"}), \
            patch.object(strategy, "get_fomc_calendar", return_value={}), \
            _stub_valet(_all_series_payload()):
        result = strategy.analyze_macro_context()

    assert result["canada_source"] == "Bank of Canada"
    # The FRED sentinel (9.99) must be nowhere in the output.
    assert result["key_indicators"]["BoC Rate (CA)"] == "2.25%"
    assert result["key_indicators"]["Canada CPI"] == "1.85%"
    assert "9.99" not in str(result)
    assert result["boc_detail"]["last_change"]["change_bps"] == -25
    assert "Bank of Canada" in result["canadian_strategy"]


def test_macro_strategy_falls_back_to_fred_when_boc_is_down():
    import tools.macro_strategy as strategy

    us_macro = {
        "fed_funds": {"current_rate": "3.63%"},
        "inflation": {"headline_inflation": "2.4%"},
        "unemployment": {"current_rate": "4.3%"},
        "treasury_yields": {"10_year_yield": "4.44%", "2_year_yield": "4.14%"},
        "gdp": {"trend": "Stable"},
    }
    with patch.object(strategy, "get_all_macro_indicators", return_value=us_macro), \
            patch.object(strategy, "get_systemic_risk_indicators",
                         return_value={"crash_risk": "Low", "liquidity_status": "Expanding"}), \
            patch.object(strategy, "get_canada_metrics",
                         return_value={"interest_rate": "3.75%", "inflation": "2.1%"}), \
            patch.object(strategy, "get_fomc_calendar", return_value={}), \
            _stub_valet(None, err="Bank of Canada Valet unreachable: ConnectionError"):
        result = strategy.analyze_macro_context()

    assert result["canada_source"] == "FRED (OECD re-publication)"
    assert result["key_indicators"]["BoC Rate (CA)"] == "3.75%"
    assert "boc_detail" not in result
