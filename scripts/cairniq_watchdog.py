#!/usr/bin/env python3
"""CairnIQ watchdog — external kill-switch AND liveness supervisor.

Runs periodically (launchd com.cairniq.watchdog, every ~2 min). It is independent
of the server process, so it still fires when the server is looping. It checks:

  1. LLM HARD budget (agent/llm_budget.over_hard_budget) — persistent, restart-safe.
  2. Server restart-storm — a jump in uvicorn "Started server process" lines in
     logs/cairniq.stderr.log since the previous run.
  3. Liveness — is anything actually listening on the server port?
  4. SUPERVISION — is the thing listening actually owned by launchd?
  5. SERVING — does the bound, supervised port actually answer a request?

On a runaway breach (1 or 2) it ALERTS (log + alert file + best-effort macOS
notification) and, unless disabled, DISABLES the server LaunchAgent (`launchctl
bootout`) so a runaway stops and STAYS down for inspection — rather than being
respawned by KeepAlive.

Liveness (3) covers the opposite failure: the service silently staying DOWN. If a
server is started outside launchd it wins the port race, launchd's own start hits
the single-instance guard in server.py and exits 0, and KeepAlive
{SuccessfulExit: false} then leaves the job dormant — so when that unsupervised
process dies, nothing brings the service back. Here we detect "nothing listening +
job loaded but idle" and `launchctl kickstart` it.

Supervision (4) covers the state BETWEEN those two, which the liveness probe used
to score as perfectly healthy: the port is bound, but by a process launchd does
not own. Everything looks fine — the port answers, health checks pass — while the
service is running with no supervision at all, so the moment it dies it stays
dead. It also makes deploys lie: `launchctl kickstart -k` cannot replace an
instance launchd does not own, so a fresh start exits on the single-instance
guard and the OLD CODE keeps serving. Observed live on cairniq 2026-07-25.
Detection is unconditional and alerts once; ADOPTION (SIGTERM the orphan, then
kickstart into the freed port) is opt-in, because killing a process that is
serving correctly is a bigger blast radius than the latent risk it removes.

Serving (5) closes the last of these, and it is the one that cost the most. Checks
3 and 4 both ask who holds the port; neither ever sent a request. `/context`
returned HTTP 500 eight times over roughly five hours on cairniq while this script
logged `ok` every two minutes, because the box genuinely was up — a half-deploy
wrote new templates with no restart, and Jinja re-reads templates per request
while Python stays frozen at exec. So a GET now goes to /api/health and its status
code is recorded, and the endpoint reports whether the running process predates the
newest source on disk, which is that half-deploy's signature before it renders as
a 500 on some page nobody probed. Neither state restarts anything from here: the
process is running and this watchdog does not kill running processes.

Revival is deliberately conservative so it can never fight the kill-switch above:
  * never in the same run as a runaway breach,
  * never while the watchdog itself disabled the job (`disabled_at` state key),
  * only when launchd reports the job loaded with NO running pid (a job that is
    running but not yet listening is just booting — we wait, we never kill it),
  * only after DOWN_CONFIRMATIONS consecutive down checks,
  * at most once per REVIVE_COOLDOWN_S, and at most REVIVE_MAX_TRIES times before
    it gives up and only alerts.
  The cooldown caps our contribution to ~1 restart per 10 min, well under the
  restart-storm threshold, so the two halves cannot chase each other.

Logging: this script writes to STDOUT ONLY. The launchd plist redirects both
StandardOutPath and StandardErrorPath to logs/cairniq.watchdog.log — do not also
write to that file from here, or every line lands in it twice.

Toggles (user_data/.env or launchd env):
  AIDLC_WATCHDOG_AUTOKILL=0            disable auto-disable (alert only)
  AIDLC_WATCHDOG_MAX_RESTARTS=8        restart-storm threshold per interval
  AIDLC_WATCHDOG_AUTOREVIVE=0          disable liveness kickstart (alert only)
  AIDLC_WATCHDOG_ADOPT_ORPHAN=1        hand an unsupervised port back to launchd
                                       (default 0 = detect and alert only)
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STDERR_LOG = os.path.join(ROOT, "logs", "cairniq.stderr.log")
STATE_PATH = os.path.join(ROOT, "user_data", ".watchdog_state.json")
ALERT_PATH = os.path.join(ROOT, "user_data", "ALERT_runaway.txt")
AGENT_LABEL = "com.cairniq.server"
AGENT_PLIST = os.path.expanduser(f"~/Library/LaunchAgents/{AGENT_LABEL}.plist")

# Liveness probe. Mirrors the single-instance guard in server.py: a bind of
# "0.0.0.0" is still probed over loopback.
SERVER_HOST = os.environ.get("CAIRNIQ_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("PORT", "8000"))
# The serving probe (7.1 Step 3). A bound port is not a working surface.
HEALTH_PATH = "/api/health"
HTTP_TIMEOUT_S = 5.0
HEALTH_READ_LIMIT = 16384  # a health payload is tiny; never slurp an error page

# LOG VOCABULARY — a contract, not prose. `tools.availability` greps the probe log
# for these markers to measure the 5xx axis, so changing either string is a
# breaking change to a measurement and both sides must move together.
SERVING_ERROR_MARKER = "SERVING-ERRORS HTTP"
STALE_CODE_MARKER = "STALE-CODE:"

# The HEALTHY line, and it is load-bearing for the measurement rather than for
# operations. A bare `ok` is what the OLD tcp-only probe wrote, so a log full of
# them is not evidence that anything was ever requested — which left
# `serving_probe_active` able to turn true only when something BROKE, and a
# permanently healthy host reporting its working probe as absent. Recording the
# status code on the good path is what makes the probe's presence falsifiable.
HEALTHY_MARKER = "ok HTTP"

DOWN_CONFIRMATIONS = 2  # consecutive down checks before we act (~4 min)
STARTING_GRACE_CHECKS = 5  # checks a booting server may hold a pid without listening
REVIVE_COOLDOWN_S = 600  # never kickstart more than once per 10 min
REVIVE_MAX_TRIES = 3  # then alert and stop trying


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(msg: str) -> None:
    # stdout only — launchd owns the log file (see module docstring).
    print(f"{_now()} {msg}", flush=True)


def _load_state() -> dict:
    try:
        with open(STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(d: dict) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=2)
    except OSError:
        pass


def _count_starts() -> int:
    """Total 'Started server process' lines seen so far in the stderr log."""
    try:
        with open(STDERR_LOG, encoding="utf-8", errors="ignore") as fh:
            return sum(1 for ln in fh if "Started server process" in ln)
    except OSError:
        return 0


def _notify(title: str, body: str) -> None:
    """Best-effort desktop notification (visible if someone is at the cairniq
    screen). Replace/extend with email or push for remote alerting."""
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification {json.dumps(body)} with title {json.dumps(title)}'],
            timeout=5, capture_output=True,
        )
    except Exception:
        pass


def _raise_alert(reason: str) -> None:
    msg = f"RUNAWAY DETECTED: {reason}"
    log(msg)
    try:
        with open(ALERT_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {msg}\n")
    except OSError:
        pass
    _notify("CairnIQ runaway", reason)


def _disable_server() -> bool:
    uid = os.getuid()
    # Prefer modern bootout; fall back to legacy unload.
    for cmd in (
        ["launchctl", "bootout", f"gui/{uid}/{AGENT_LABEL}"],
        ["launchctl", "unload", "-w", AGENT_PLIST],
    ):
        try:
            r = subprocess.run(cmd, timeout=10, capture_output=True, text=True)
            if r.returncode == 0:
                log(f"Disabled {AGENT_LABEL} via: {' '.join(cmd)}")
                return True
        except Exception as e:  # noqa: BLE001
            log(f"disable attempt failed ({' '.join(cmd)}): {e}")
    log(f"WARNING: could not disable {AGENT_LABEL}; manual intervention needed.")
    return False


# ── Liveness ────────────────────────────────────────────────────────────────

def _probe_host() -> str:
    # "0.0.0.0" here is only a comparison (to probe via loopback), not a bind.
    return "127.0.0.1" if SERVER_HOST in ("0.0.0.0", "") else SERVER_HOST  # noqa: S104  # nosec B104


def _port_open() -> bool:
    """True if something is listening on the server port.

    A bare TCP connect, and deliberately still that: `_adopt_orphan` needs to know
    when the port is FREE, which is a socket question and not an HTTP one. What it
    must no longer be used for on its own is deciding health — see `_check_serving`.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(2.0)
            return probe.connect_ex((_probe_host(), SERVER_PORT)) == 0
    except OSError:
        return False


def _http_status(path: str = HEALTH_PATH) -> tuple[int | None, dict]:
    """`(http_status, parsed_json)` for a GET against the local server.

    `(None, {})` means the request itself failed — the port answered a connect but
    not an HTTP exchange. Loopback plain HTTP, so `urllib` is correct here and
    needs no CA bundle (the framework Python on cairniq has no cafile, which makes
    raw urllib HTTPS fail — not a concern for http://127.0.0.1).

    An HTTP error status is a RESULT, not an exception: `HTTPError` is caught and
    its code returned, because a 500 is precisely the thing this probe exists to
    see.
    """
    import urllib.error
    import urllib.request

    url = f"http://{_probe_host()}:{SERVER_PORT}{path}"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as resp:  # noqa: S310
            body = resp.read(HEALTH_READ_LIMIT)
            try:
                return resp.status, json.loads(body or b"{}")
            except (ValueError, TypeError):
                return resp.status, {}
    except urllib.error.HTTPError as e:
        # The load-bearing branch. 4xx and 5xx both land here.
        try:
            payload = json.loads(e.read(HEALTH_READ_LIMIT) or b"{}")
        except Exception:  # noqa: BLE001
            payload = {}
        return e.code, payload
    except Exception:  # noqa: BLE001 — a probe must never raise into the caller
        return None, {}


def _check_serving(state: dict) -> str:
    """Does the bound port actually SERVE, or is it up and returning errors?

    THE GAP THIS CLOSES, and the reason 7.1 grew a fourth number. Every check
    above this one asks whether a process holds the port and whether launchd owns
    it. None of them ever sent a request. `/context` returned HTTP 500 eight times
    over roughly five hours on cairniq and this watchdog logged `ok` every two
    minutes throughout, because the box was up the whole time: a half-deploy wrote
    new templates with no restart, Jinja re-reads templates per request while
    Python is frozen at exec, and 2.9's new template rendered against the OLD
    `get_dashboard_context`. So availability measured as "the process is alive"
    cannot see a surface that is serving errors, and the measured coverage figure
    in 7.1 is only an UPPER BOUND until this probe exists.

    A 5xx is reported and never acted on. The watchdog's standing rule is that it
    does not kill a process which is running, and a server returning 500s still
    holds the port with a live launchd-owned pid — so the revive path above cannot
    fire on it anyway, and must not be made to. Restarting it is the operator's
    call (that is the post-receive hook decision, deliberately left open).

    Status strings are the probe log's vocabulary and `tools.availability` parses
    them, so they are a CONTRACT — see `SERVING_ERROR_MARKER`.
    """
    status, payload = _http_status()

    if status is None:
        # Bound but not completing an HTTP exchange: a hung worker, or a process
        # mid-startup that has already bound. Not revived here — the caller's
        # starting-grace path owns that state.
        n = int(state.get("not_serving_checks", 0)) + 1
        state["not_serving_checks"] = n
        log(f"port {SERVER_PORT} is bound but {HEALTH_PATH} did not complete an "
            f"HTTP exchange ({n} consecutive)")
        return "not-serving"

    state.pop("not_serving_checks", None)

    if status >= 500:
        detail = (
            f"{SERVING_ERROR_MARKER} {status} from {HEALTH_PATH} — the port is bound and "
            f"launchd owns it, but the app is serving errors. Not restarting from here; "
            f"a half-deploy (new files, no restart) looks exactly like this."
        )
        if not state.get("serving_error_alerted"):
            _raise_alert(detail)
            state["serving_error_alerted"] = True
        else:
            log(detail)
        return "serving-errors"

    if state.pop("serving_error_alerted", None):
        log(f"{HEALTH_PATH} is answering {status} again — surface recovered")

    # Carried to main() so the healthy line can name the code it actually saw.
    state["last_http_status"] = status

    # 200, but running code older than what is on disk: the half-deploy state,
    # caught BEFORE it renders as a 500 on some page nobody probed. Reported and
    # not acted on, for the same reason as above.
    if payload.get("code_stale") is True:
        log(f"{STALE_CODE_MARKER} {HEALTH_PATH} answers {status}, but the running "
            f"process started before the newest source on disk "
            f"({payload.get('code_stale_detail') or 'no detail'}) — a restart is "
            f"needed for the deployed code to actually serve")
        return "up-stale-code"

    return "up"


def _tool(name: str, *candidates: str) -> str:
    """Absolute path to a system binary, falling back to a bare PATH lookup.

    Load-bearing under launchd. A scheduled job does NOT inherit an interactive
    shell's PATH, and `subprocess.run(["lsof", ...])` raised FileNotFoundError
    on every scheduled run while working perfectly when the same script was run
    by hand over SSH — so the supervision check silently returned "unknown"
    forever and logged "ok" every two minutes. That is this codebase's signature
    failure (works when you test it, inert in production), reproduced inside the
    fix for it, and caught only by reading the launchd log rather than trusting
    the manual run.
    """
    for path in candidates:
        if os.path.exists(path):
            return path
    return name


def _listener_pid() -> int | None:
    """PID currently listening on the server port, or None if undeterminable.

    None means UNKNOWN, never "nobody" — the orphan check must not fire on a
    missing `lsof`, or a tooling gap would present as a supervision fault. That
    tolerance is why the PATH problem above was silent, so the failure is now
    logged loudly rather than absorbed.
    """
    try:
        r = subprocess.run(
            [_tool("lsof", "/usr/sbin/lsof", "/usr/bin/lsof"),
             "-nP", f"-iTCP:{SERVER_PORT}", "-sTCP:LISTEN", "-t"],
            timeout=10, capture_output=True, text=True,
        )
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: cannot determine the listening pid ({e}) — "
            f"supervision check is INERT this run, not passing")
        return None
    pids = [p for p in (r.stdout or "").split() if p.strip().isdigit()]
    if not pids:
        return None
    return int(pids[0])


def _is_descendant(child: int, ancestor: int, max_depth: int = 6) -> bool:
    """True if `child` is `ancestor` or below it in the process tree.

    Guards the orphan check against a legitimate multi-process server: if the
    listener is a worker forked from the launchd-owned parent, launchd DOES
    supervise it and it must not be reported as an orphan. Today's deployment is
    single-process (the launchd pid and the listener are the same), but a future
    `--workers` would silently turn this check into a false alarm otherwise.
    """
    if child == ancestor:
        return True
    seen, current = set(), child
    for _ in range(max_depth):
        if current in seen or current <= 1:
            return False
        seen.add(current)
        try:
            r = subprocess.run([_tool("ps", "/bin/ps"), "-o", "ppid=", "-p", str(current)],
                               timeout=5, capture_output=True, text=True)
            parent = int((r.stdout or "").strip() or 0)
        except Exception:  # noqa: BLE001
            return False
        if parent == ancestor:
            return True
        current = parent
    return False


def _adopt_orphan(listener: int, state: dict) -> str:
    """Hand the port back to launchd: stop the orphan, then start the job.

    Deliberately NOT the default. The orphan is SERVING, so this trades a real
    ~50s outage now against a latent risk (nothing restarts it if it dies). The
    watchdog's standing rule is never to kill a running instance, so adoption is
    opt-in via AIDLC_WATCHDOG_ADOPT_ORPHAN=1 and rate-limited like any revive.

    `kickstart` is issued WITHOUT -k, and only once the port is actually free:
    with the port still held, launchd's fresh instance hits server.py's
    single-instance guard, exits 0, and leaves the orphan serving — the exact
    trap that makes `kickstart -k` useless against this state.
    """
    log(f"adopting orphan pid {listener}: sending SIGTERM to free port {SERVER_PORT}")
    try:
        os.kill(listener, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as e:
        log(f"could not signal orphan {listener}: {e}")
        return "orphan-adopt-failed"

    for _ in range(15):
        time.sleep(1)
        if not _port_open():
            break
    else:
        log(f"orphan {listener} still holding {SERVER_PORT} after SIGTERM — not escalating; will retry next cycle")
        return "orphan-adopt-failed"

    state["last_revive"] = time.time()
    return "orphan-adopted" if _kickstart() else "orphan-adopt-failed"


def _job_status() -> tuple[bool, int | None]:
    """(loaded, pid) for the server LaunchAgent.

    pid is None when the job is loaded but not currently running — the dormant
    state this watchdog exists to catch.
    """
    try:
        r = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{AGENT_LABEL}"],
            timeout=10, capture_output=True, text=True,
        )
    except Exception as e:  # noqa: BLE001
        log(f"launchctl print failed: {e}")
        return (False, None)
    if r.returncode != 0:
        return (False, None)
    for line in r.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid ="):
            try:
                return (True, int(stripped.split("=", 1)[1].strip()))
            except ValueError:
                break
    return (True, None)


def _kickstart() -> bool:
    """Start the (loaded, idle) server job.

    Deliberately WITHOUT -k: we must never kill an instance that is already
    running, because "running but not listening" is the normal heavy-startup
    state, not a fault.
    """
    cmd = ["launchctl", "kickstart", f"gui/{os.getuid()}/{AGENT_LABEL}"]
    try:
        r = subprocess.run(cmd, timeout=30, capture_output=True, text=True)
    except Exception as e:  # noqa: BLE001
        log(f"kickstart failed: {e}")
        return False
    if r.returncode == 0:
        log(f"kickstarted {AGENT_LABEL}")
        return True
    log(f"kickstart failed (rc={r.returncode}): {(r.stderr or r.stdout).strip()}")
    return False


def _check_supervision(state: dict) -> str | None:
    """Is the process serving the port actually supervised by launchd?

    THE GAP THIS CLOSES. `_check_liveness` treated an open port as
    unconditionally healthy — it asked whether *something* was listening, never
    *who*. So a server started outside launchd (or one that outlived its job)
    keeps the port bound, launchd's job sits at `state = not running`, and the
    probe reports "up" forever. CairnIQ then runs with NO supervision at all: if
    that process dies, nothing restarts it and nothing notices. Observed live on
    cairniq 2026-07-25 — listener pid 599 against a job that launchd considered
    not running, and it had been that way long enough to survive a reboot cycle.

    It also breaks deploys silently: `launchctl kickstart -k` cannot replace an
    instance launchd does not own, so the fresh process exits on the
    single-instance guard and the OLD CODE keeps serving while every health
    check passes.

    Returns None when supervision is fine (caller continues its normal "up"
    path), or a status string when the port is held by an unsupervised process.
    """
    loaded, job_pid = _job_status()
    listener = _listener_pid()

    # Unknown is not a fault: no lsof, or a job we cannot read, must not raise a
    # supervision alarm. Same rule as everywhere else in this codebase.
    if listener is None or not loaded:
        state.pop("orphan_since", None)
        return None

    if job_pid is not None and _is_descendant(listener, job_pid):
        if state.pop("orphan_since", None):
            log(f"port {SERVER_PORT} is back under launchd supervision (pid {job_pid})")
        state.pop("orphan_alerted", None)
        return None

    first_seen = state.get("orphan_since")
    if not first_seen:
        state["orphan_since"] = _now()
        first_seen = state["orphan_since"]

    detail = (
        f"port {SERVER_PORT} is served by pid {listener}, which launchd does not own "
        f"(job pid = {job_pid}). If it dies nothing will restart it, and "
        f"`launchctl kickstart -k` will not replace it — a deploy would leave the old code "
        f"serving while health checks pass. First seen {first_seen}."
    )

    if os.environ.get("AIDLC_WATCHDOG_ADOPT_ORPHAN", "0") == "1":
        last = float(state.get("last_revive", 0) or 0)
        remaining = REVIVE_COOLDOWN_S - (time.time() - last)
        if remaining > 0:
            log(f"orphaned listener on {SERVER_PORT}; adopt in cooldown ({int(remaining)}s left)")
            return "orphaned"
        log(f"ORPHANED SERVER: {detail}")
        result = _adopt_orphan(listener, state)
        if result == "orphan-adopted":
            state.pop("orphan_since", None)
            state.pop("orphan_alerted", None)
        return result

    # Default: report, do not act. Killing a process that is serving correctly is
    # a bigger blast radius than the latent risk it removes, so the trade is the
    # operator's to make (AIDLC_WATCHDOG_ADOPT_ORPHAN=1).
    if not state.get("orphan_alerted"):
        _raise_alert(f"server is UNSUPERVISED — {detail}")
        _notify("CairnIQ unsupervised", f"pid {listener} holds port {SERVER_PORT} outside launchd")
        state["orphan_alerted"] = True
    else:
        log(f"still unsupervised: {detail}")
    return "orphaned"


def _check_liveness(state: dict) -> str:
    """Probe the server port and revive a dormant LaunchAgent.

    Returns a status string: up | starting | hung | down | revived |
    kickstart-failed | gave-up | disabled | unloaded.
    """
    autorevive = os.environ.get("AIDLC_WATCHDOG_AUTOREVIVE", "1") != "0"
    where = f"{_probe_host()}:{SERVER_PORT}"

    if _port_open():
        orphan_status = _check_supervision(state)
        if orphan_status:
            return orphan_status
        if state.pop("disabled_at", None):
            log(f"server is up on {where} again — clearing watchdog disable flag")
        # Pop both before testing — `or` would short-circuit and strand one of
        # them, letting revive_tries accumulate across unrelated outages until
        # the give-up threshold trips on a healthy system.
        was_down = state.pop("down_checks", None)
        had_tried = state.pop("revive_tries", None)
        # last_revive is deliberately NOT cleared: it rate-limits kickstarts
        # globally, so a server that flaps (up a minute, down a minute) still
        # gets at most one kickstart per cooldown instead of one per cycle.
        if was_down or had_tried:
            log(f"server recovered on {where}")
        state.pop("starting_checks", None)
        state.pop("gave_up_alerted", None)
        # The port is bound and launchd owns it. That used to be the end of the
        # check and was where the five-hour 500s outage hid — now ask whether it
        # actually serves.
        return _check_serving(state)

    loaded, pid = _job_status()

    # The watchdog itself took the job down for inspection — leave it down.
    disabled_at = state.get("disabled_at")
    if disabled_at:
        if not loaded:
            log(
                f"server down on {where}, but watchdog disabled {AGENT_LABEL} at {disabled_at} "
                f"for inspection — NOT reviving. Re-enable with: "
                f"launchctl bootstrap gui/$(id -u) {AGENT_PLIST}"
            )
            return "disabled"
        # Someone re-loaded it by hand; resume normal supervision.
        log(f"{AGENT_LABEL} was re-loaded by hand (disabled {disabled_at}) — resuming supervision")
        state.pop("disabled_at", None)

    if not loaded:
        log(
            f"WARNING: nothing listening on {where} and {AGENT_LABEL} is not loaded in launchd. "
            f"Load it with: launchctl bootstrap gui/$(id -u) {AGENT_PLIST}"
        )
        return "unloaded"

    if pid is not None:
        # Running but not serving yet: CairnIQ's startup is heavy (ticker
        # downloads, agent init). Wait it out; never kill it from here.
        n = int(state.get("starting_checks", 0)) + 1
        state["starting_checks"] = n
        if n >= STARTING_GRACE_CHECKS:
            log(
                f"WARNING: {AGENT_LABEL} has held pid {pid} for {n} checks without listening on "
                f"{where} — possible hung startup. Not killing it; inspect manually."
            )
            return "hung"
        log(f"not listening on {where} yet; {AGENT_LABEL} running (pid {pid}) — startup in progress ({n}/{STARTING_GRACE_CHECKS})")
        return "starting"

    state.pop("starting_checks", None)

    # Loaded, no pid: dormant. This is the failure this probe exists for.
    down = int(state.get("down_checks", 0)) + 1
    state["down_checks"] = down
    if down < DOWN_CONFIRMATIONS:
        log(f"nothing listening on {where}; {AGENT_LABEL} loaded but idle (confirmation {down}/{DOWN_CONFIRMATIONS})")
        return "down"

    tries = int(state.get("revive_tries", 0))
    if tries >= REVIVE_MAX_TRIES:
        if not state.get("gave_up_alerted"):
            _raise_alert(
                f"server DOWN on {where}: {REVIVE_MAX_TRIES} kickstart attempts failed — manual intervention needed"
            )
            state["gave_up_alerted"] = True
        else:
            log(f"server still down on {where}; gave up after {REVIVE_MAX_TRIES} attempts — manual intervention needed")
        return "gave-up"

    last = float(state.get("last_revive", 0) or 0)
    remaining = REVIVE_COOLDOWN_S - (time.time() - last)
    if remaining > 0:
        log(f"server down on {where}; in revive cooldown ({int(remaining)}s left)")
        return "down"

    if not autorevive:
        log(f"server down on {where} and {AGENT_LABEL} is idle — AUTOREVIVE disabled, not kickstarting")
        return "down"

    state["last_revive"] = time.time()
    state["revive_tries"] = tries + 1
    log(f"server down on {where} and {AGENT_LABEL} is loaded but idle — kickstarting (attempt {tries + 1}/{REVIVE_MAX_TRIES})")
    return "revived" if _kickstart() else "kickstart-failed"


def main() -> int:
    autokill = os.environ.get("AIDLC_WATCHDOG_AUTOKILL", "1") != "0"
    max_restarts = int(float(os.environ.get("AIDLC_WATCHDOG_MAX_RESTARTS", "8")))

    state = _load_state()
    reasons = []

    # 1. LLM hard budget
    try:
        from agent import llm_budget
        hard = llm_budget.over_hard_budget()
        st = llm_budget.status()
        if hard:
            reasons.append(f"LLM hard budget: {hard} | {st}")
    except Exception as e:  # noqa: BLE001
        log(f"budget check error: {e}")

    # 2. Restart-storm
    starts = _count_starts()
    prev_starts = int(state.get("starts", starts))
    delta = starts - prev_starts
    if delta < 0:  # log rotated/truncated
        delta = 0
    if delta >= max_restarts:
        reasons.append(f"restart-storm: {delta} server starts since last check (>= {max_restarts})")
    state["starts"] = starts
    state["checked"] = _now()

    if reasons:
        for r in reasons:
            _raise_alert(r)
        if autokill:
            if _disable_server():
                # Remember that WE took it down, so the liveness probe below
                # never resurrects a runaway on the next run.
                state["disabled_at"] = _now()
        else:
            log("AUTOKILL disabled — alert only.")
        _save_state(state)
        return 1

    # 3. Liveness — only when nothing is running away, so the two halves of this
    #    watchdog can never fight each other.
    status = _check_liveness(state)
    _save_state(state)

    if status == "up":
        # `ok HTTP 200`, not a bare `ok`. The code is what proves a request was
        # actually made; the old probe wrote `ok` having only opened a socket.
        code = state.get("last_http_status")
        log(f"{HEALTHY_MARKER} {code}" if code else "ok")
        return 0
    # "up-stale-code" is deliberately absent from both branches' happy path: the
    # surface answers, so it is not a failure exit, but it must never log "ok" —
    # that line is what a half-deploy hid behind for five hours.
    return 1 if status in ("down", "hung", "gave-up", "kickstart-failed", "unloaded",
                       "orphaned", "orphan-adopt-failed", "serving-errors",
                       "not-serving") else 0


if __name__ == "__main__":
    raise SystemExit(main())
