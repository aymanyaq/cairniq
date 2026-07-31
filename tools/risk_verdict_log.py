"""
RiskManager verdict audit trail (Advisor Roadmap Theme 2.1).

Persists every parsed Risk Judge verdict — score, violations, retry outcome —
to the per-profile `risk_verdicts.jsonl`. Until now verdicts existed only in
transient component logs; this file is the durable audit trail and the
calibration corpus that Theme 2's other items (IPS pre-check thresholds,
golden-set eval harness) and Theme 1.4's weekly self-review read from.

One JSON object per line. Two record shapes, distinguished by `event`:
  - "verdict":  a full parsed judge pass (score, risk_result, violations,
                retry_count/retry_outcome, verdict_text)
  - "bypassed": the judge was skipped because a risk assessment already
                existed in the current turn (DeepReasoning embedded one)
"""
import json
import os
from datetime import datetime
from typing import Any

from tools.user_profile import get_data_path

_VERDICT_FILENAME = "risk_verdicts.jsonl"

# Caps keep one verbose or malformed pass from bloating the audit trail: the
# judge prompt limits verdicts to ~200 words, so 4000 chars only truncates
# pathological output; 400 chars of query is enough to identify the prompt
# (quick-action lens headers sit in the first line).
_MAX_TEXT_CHARS = 4000
_MAX_QUERY_CHARS = 400


def _verdict_file() -> str:
    return get_data_path(_VERDICT_FILENAME)


def log_risk_verdict(record: dict[str, Any]) -> bool:
    """
    Append one verdict record as a JSON line, stamping `ts`.

    Never raises: the audit trail is strictly best-effort and must not be able
    to break the risk gate it documents. Returns True when the line was written.
    """
    try:
        entry = dict(record)
        entry["ts"] = datetime.now().isoformat(timespec="seconds")
        if isinstance(entry.get("verdict_text"), str):
            entry["verdict_text"] = entry["verdict_text"][:_MAX_TEXT_CHARS]
        if isinstance(entry.get("query"), str):
            entry["query"] = entry["query"][:_MAX_QUERY_CHARS]
        with open(_verdict_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return True
    except Exception as e:
        try:
            from agent.utils import safe_print

            safe_print(f"⚠️ risk_verdicts.jsonl write failed: {e}")
        except Exception:
            pass
        return False


def get_recent_verdicts(limit: int = 50, include_bypassed: bool = False) -> list[dict[str, Any]]:
    """
    Return the last `limit` verdict records, oldest → newest.

    Skips corrupt lines so one bad write can't poison the whole corpus, and by
    default filters out "bypassed" markers so calibration consumers see only
    real judged passes. Returns [] when no trail exists yet.
    """
    try:
        path = _verdict_file()
        if not os.path.exists(path):
            return []
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if not include_bypassed and rec.get("event") == "bypassed":
                    continue
                records.append(rec)
        return records[-limit:] if limit and limit > 0 else records
    except Exception:
        return []
