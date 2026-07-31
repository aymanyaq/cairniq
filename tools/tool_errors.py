"""Canonical degraded-result contract for data tools.

When a tool cannot deliver real data (missing API key, exhausted rate limits,
upstream outage), it must say so explicitly instead of returning ``{}``, ``[]``
or ``None``. A silent empty flows into the LLM context looking like "I checked
and found nothing", and the model will confidently narrate an analysis over
hollow data. Returning :func:`unavailable` gives the model (and the user) the
reason, so the response can say "insider data was unavailable: FMP_API_KEY not
configured" instead.

Convention:
- Genuine "no data exists" (e.g. no insider trades this month) keeps the
  natural empty shape — that IS a real answer.
- "I could not check" returns ``unavailable(source, reason)``.
"""
from typing import Any

UNAVAILABLE = "unavailable"


def unavailable(source: str, reason: str, **extra: Any) -> dict[str, Any]:
    """Build the canonical degraded-result payload.

    Args:
        source: The data provider or subsystem that failed (e.g. "FMP").
        reason: Human-readable cause, actionable where possible
                (e.g. "FMP_API_KEY not configured — add it in Settings → API Keys").
        extra: Optional additional context merged into the payload.
    """
    payload: dict[str, Any] = {"status": UNAVAILABLE, "source": source, "reason": reason}
    payload.update(extra)
    return payload


def is_unavailable(value: Any) -> bool:
    """True if ``value`` is a degraded-result payload from :func:`unavailable`."""
    return isinstance(value, dict) and value.get("status") == UNAVAILABLE


def missing_key_reason(env_var: str) -> str:
    """Standard reason string for an unconfigured API key."""
    return f"{env_var} not configured — add it in Settings → API Keys to enable this data source."
