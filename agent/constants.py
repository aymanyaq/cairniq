"""Shared agent-level constants.

Single source of truth for values referenced from multiple modules
(e.g. graph routing and supervisor routing) so the copies can never drift.
"""

# Queries matching any of these keywords are system/health diagnostics, not
# financial advice — they bypass the RiskManager compliance gate.
# Used by agent/graph.py (after_deep_reasoning) and agent/nodes/supervisor.py.
HEALTH_CHECK_KEYWORDS = (
    "health check",
    "run_health_check",
    "diagnostics",
    "system diagnostics",
    "health check protocol",
    "system health",
    "portfolio integrity",
)
