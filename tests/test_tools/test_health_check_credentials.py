"""Tests for credential classification in the tool health check.

The key behaviour: optional data-provider keys (FMP, Alpha Vantage, FRED, etc.)
have graceful fallbacks (yfinance, web scraping), so an unset key must NEVER be
reported as a missing prerequisite. Only the active LLM provider's credentials
are required. Regression guard for the "⚠️ Missing Credentials" false alarm.
"""

from tools.health_check import (
    _OPTIONAL_DATA_KEYS,
    _classify_credentials,
)

# A clean environment with no credentials at all.
_EMPTY_ENV: dict[str, str] = {}


def _missing_names(missing_results):
    return {r["tool"].replace("Prereq: ", "") for r in missing_results}


def test_optional_keys_never_counted_as_missing():
    # Azure provider with its required creds set, but NO optional data keys.
    env = {
        "LLM_PROVIDER": "azure",
        "AZURE_OPENAI_API_KEY": "k",
        "AZURE_OPENAI_ENDPOINT": "https://x.openai.azure.com/openai/v1",
    }
    prerequisites, missing_results, optional_not_configured = _classify_credentials(env)

    # No required key is missing → no missing-prereq results at all.
    assert missing_results == []
    # Every optional key is reported as optional, not missing.
    assert set(optional_not_configured) == set(_OPTIONAL_DATA_KEYS)
    for key in _OPTIONAL_DATA_KEYS:
        assert prerequisites[key].startswith("➖ Optional")
        assert "❌" not in prerequisites[key]


def test_missing_prereqs_filter_excludes_optional():
    # Mirror the run_tool_health_check filter: only ❌ entries are "missing".
    prerequisites, _missing_results, _optional = _classify_credentials(
        {"LLM_PROVIDER": "azure", "AZURE_OPENAI_API_KEY": "k", "AZURE_OPENAI_ENDPOINT": "e"}
    )
    missing_prereqs = [k for k, v in prerequisites.items() if "❌" in v]
    assert missing_prereqs == []  # optional keys (➖) must not appear


def test_active_provider_required_keys_flagged_when_unset():
    # Azure selected but credentials absent → those (and only those) are missing.
    prerequisites, missing_results, optional_not_configured = _classify_credentials(
        {"LLM_PROVIDER": "azure"}
    )
    assert _missing_names(missing_results) == {"AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"}
    assert prerequisites["AZURE_OPENAI_API_KEY"].startswith("❌")
    # Optional keys are still only optional, never missing.
    assert "FMP_API_KEY" in optional_not_configured
    assert prerequisites["FMP_API_KEY"].startswith("➖")


def test_provider_specific_required_set():
    # Switching the provider changes which credentials are required (vendor-neutral).
    for provider, expected in [
        ("bedrock", {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}),
        ("openai", {"OPENAI_API_KEY"}),
        ("anthropic", {"ANTHROPIC_API_KEY"}),
        ("google", {"GOOGLE_API_KEY"}),
        ("azure", {"AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"}),
    ]:
        _prereq, missing_results, _opt = _classify_credentials({"LLM_PROVIDER": provider})
        assert _missing_names(missing_results) == expected


def test_set_optional_key_marked_set_not_optional():
    env = {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "k", "FMP_API_KEY": "fmp"}
    prerequisites, missing_results, optional_not_configured = _classify_credentials(env)
    assert missing_results == []
    assert prerequisites["FMP_API_KEY"] == "✅ Set"
    assert "FMP_API_KEY" not in optional_not_configured
