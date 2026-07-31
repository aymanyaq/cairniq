"""
Persistent alerts inbox (Advisor Roadmap Theme 3.2).

The delivery rail for the "advisor who calls first" arc: regime flips,
action-required signals, catalyst escalations, watch-condition hits (3.3),
the intraday sentinel (3.4), and decision proposals (3.8) all land here.

Per-profile JSONL store (`alerts.jsonl`). One JSON object per line:
  {id, ts, severity, title, message, source, read, dedup_key, data}

Delivery on raise:
  - WebSocket broadcast ({"type": "alert", "data": record}) over the existing
    ConnectionManager, thread-safe via the captured main loop — same pattern
    as broadcast_graph_update.
  - macOS desktop notification for severity >= warning (watchdog's proven
    osascript pattern), so an alert still reaches a user with no tab open.

Dedup: a raised alert whose dedup_key matches an existing UNREAD alert
refreshes that record (ts + message) instead of duplicating it — a sentinel
that fires every tick must not bury the inbox.

Read state: JSONL is append-only, so mark_read rewrites the file atomically
(tmp + replace). The store is capped at _MAX_RECORDS, so a rewrite is cheap.
"""
import asyncio
import json
import os
import subprocess
import uuid
from datetime import datetime
from typing import Any

from agent.logger import log_to_component
from tools.user_profile import get_data_path

_ALERTS_FILENAME = "alerts.jsonl"
_MAX_RECORDS = 500
_MAX_MESSAGE_CHARS = 2000
SEVERITIES = ("info", "warning", "critical")
_NOTIFY_MIN_SEVERITY = "warning"


def _alerts_file() -> str:
    return get_data_path(_ALERTS_FILENAME)


def _severity_rank(sev: str) -> int:
    try:
        return SEVERITIES.index(sev)
    except ValueError:
        return 0


def _load_all() -> list[dict[str, Any]]:
    """Read every alert, oldest → newest. Skips corrupt lines; [] on any error."""
    try:
        path = _alerts_file()
        if not os.path.exists(path):
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("id"):
                    records.append(rec)
        return records
    except Exception:
        return []


def _write_all(records: list[dict[str, Any]]) -> bool:
    """Atomically replace the store (tmp + replace). Never raises."""
    try:
        path = _alerts_file()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records[-_MAX_RECORDS:]:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, path)
        return True
    except Exception as e:
        log_to_component("server", "Alerts", f"alerts.jsonl rewrite failed: {e}", level=30)
        return False


def _notify_desktop(title: str, body: str) -> None:
    """macOS notification via the watchdog's proven osascript pattern."""
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification {json.dumps(body[:250])} with title {json.dumps(title[:100])}'],
            timeout=5, capture_output=True,
        )
    except Exception:
        pass


def _broadcast(record: dict[str, Any]) -> None:
    """Push the alert to every open WebSocket, safe from any thread."""
    try:
        from api import dependencies

        msg = {"type": "alert", "data": record}
        manager = dependencies.get_connection_manager()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(manager.broadcast(msg))
            return
        except RuntimeError:
            pass
        main_loop = dependencies._main_loop
        if main_loop is not None and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(manager.broadcast(msg), main_loop)
    except Exception:
        pass  # delivery must never break the producer


def raise_alert(
    title: str,
    message: str,
    severity: str = "info",
    source: str = "",
    dedup_key: str | None = None,
    data: dict[str, Any] | None = None,
    notify: bool | None = None,
) -> dict[str, Any] | None:
    """
    Raise an alert: persist to the per-profile inbox, broadcast over WebSocket,
    and (severity >= warning) post a macOS notification. Never raises.

    `dedup_key` collapses repeats: a matching UNREAD alert is refreshed
    (new ts + message) rather than duplicated. Returns the record written
    (or the refreshed record), None on failure.
    """
    try:
        if severity not in SEVERITIES:
            severity = "info"
        records = _load_all()

        if dedup_key:
            for rec in records:
                if rec.get("dedup_key") == dedup_key and not rec.get("read"):
                    rec["ts"] = datetime.now().isoformat(timespec="seconds")
                    rec["title"] = title[:200]
                    rec["message"] = message[:_MAX_MESSAGE_CHARS]
                    rec["severity"] = severity
                    if data is not None:
                        rec["data"] = data
                    rec["refreshed_count"] = rec.get("refreshed_count", 0) + 1
                    if _write_all(records):
                        _broadcast(rec)
                        return rec
                    return None

        record = {
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now().isoformat(timespec="seconds"),
            "severity": severity,
            "title": title[:200],
            "message": message[:_MAX_MESSAGE_CHARS],
            "source": source,
            "read": False,
            "dedup_key": dedup_key,
            "data": data or {},
        }
        records.append(record)
        if not _write_all(records):
            return None

        _broadcast(record)
        should_notify = notify if notify is not None else _severity_rank(severity) >= _severity_rank(_NOTIFY_MIN_SEVERITY)
        if should_notify:
            _notify_desktop(f"CairnIQ — {title}", message)
        return record
    except Exception as e:
        log_to_component("server", "Alerts", f"raise_alert failed: {e}", level=30)
        return None


def get_alerts(limit: int = 50, unread_only: bool = False) -> list[dict[str, Any]]:
    """Newest-first list of alerts. Returns [] on any error."""
    records = _load_all()
    if unread_only:
        records = [r for r in records if not r.get("read")]
    records.reverse()
    return records[:limit] if limit and limit > 0 else records


def get_unread_count() -> int:
    return sum(1 for r in _load_all() if not r.get("read"))


def get_delivery_latency() -> dict[str, Any]:
    """7.1 number 4 — how long an alert waits between being raised and being read.

    The last of 7.1's four numbers, and the only one that measures whether Theme
    3's push layer actually reaches a human. Availability answers whether the box
    was up to send; this answers whether anyone was there.

    PER-PROFILE, unlike `tools.availability`, which is deliberately global because
    a host is either up or it is not. An inbox is one person's, so this cannot ride
    on that report and is not called from it — see the note on number 4 there.

    Two exclusions, and both are the difference between a figure and a decoration:

      * **`read_via: "all"` is counted but never timed.** One mark-all click
        stamps every unread alert with the same instant, so including them would
        report N attentive reads simultaneously and quote the oldest alert's age
        as a reading time. They are reported as `bulk_read` so the ratio is
        visible: a high one means this measurement is mostly watching a button.
      * **Alerts read before `read_at` existed are `unmeasurable`.** They carry
        `read: True` and no clock. Their exclusion is reported rather than
        silently shrinking the denominator, the same rule the serving probe
        follows about the span that predates it.

    Never raises.
    """
    records = _load_all()
    read = [r for r in records if r.get("read")]
    timed: list[float] = []
    bulk = 0
    unmeasurable = 0

    for rec in read:
        raised, seen = rec.get("ts"), rec.get("read_at")
        if not seen:
            unmeasurable += 1
            continue
        if rec.get("read_via") == "all":
            bulk += 1
            continue
        try:
            delta = (datetime.fromisoformat(seen) - datetime.fromisoformat(raised)).total_seconds()
        except (TypeError, ValueError):
            unmeasurable += 1
            continue
        # A negative delta means the clock moved or a record was hand-edited;
        # it is not a zero-second read and must not be averaged as one.
        if delta < 0:
            unmeasurable += 1
            continue
        timed.append(delta)

    out: dict[str, Any] = {
        "alerts_total": len(records),
        "alerts_read": len(read),
        "alerts_unread": len(records) - len(read),
        "timed_reads": len(timed),
        "bulk_read": bulk,
        "unmeasurable_reads": unmeasurable,
    }
    if not timed:
        out["status"] = "no_data"
        out["note"] = (
            "No individually-read alert carries a read_at stamp yet, so latency is "
            "UNKNOWN rather than good. "
            + (f"{bulk} alert(s) were mark-all'd and are counted, not timed. "
               if bulk else "")
            + (f"{unmeasurable} were read before the stamp existed. "
               if unmeasurable else "")
        ).strip()
        return out

    ordered = sorted(timed)
    out["status"] = "measured"
    out["median_seconds"] = round(ordered[len(ordered) // 2], 1)
    out["p90_seconds"] = round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 1)
    out["max_seconds"] = round(ordered[-1], 1)
    out["note"] = (
        f"Median {out['median_seconds'] / 60:.1f} min to read over {len(timed)} "
        f"individually-read alert(s)"
        + (f"; {bulk} mark-all'd and excluded from the timing" if bulk else "")
        + (f"; {unmeasurable} predate the stamp" if unmeasurable else "")
        + "."
    )
    return out


def mark_read(alert_ids: list[str] | None = None, all_alerts: bool = False) -> int:
    """Mark alerts read. `all_alerts=True` marks everything; otherwise only the
    given ids. Returns how many records changed. Never raises.

    Stamps `read_at` and `read_via` (7.1 number 4). `read` on its own is a
    boolean with no clock, so there was nothing to difference against `ts` and
    delivery latency was not computable — an alert could have been read in ten
    seconds or noticed nine days later and the store said the same word.

    `read_via` is the load-bearing half, and it is why this is not just a
    timestamp. A mark-all click writes the same `read: True` to forty unread
    alerts at once, so a latency series built from `read_at` alone would report
    forty attentive reads at one instant and quote the age of the OLDEST as a
    reading time. Recording the manner is what lets the measurement keep the two
    apart instead of averaging a real signal with a single button press.
    """
    try:
        records = _load_all()
        ids = set(alert_ids or [])
        changed = 0
        stamp = datetime.now().isoformat(timespec="seconds")
        via = "all" if all_alerts else "id"
        for rec in records:
            if rec.get("read"):
                continue
            if all_alerts or rec.get("id") in ids:
                rec["read"] = True
                rec["read_at"] = stamp
                rec["read_via"] = via
                changed += 1
        if changed:
            _write_all(records)
        return changed
    except Exception:
        return 0
