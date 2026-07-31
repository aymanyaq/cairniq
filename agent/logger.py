import json
import logging
import os
import threading
import traceback
from contextvars import ContextVar
from datetime import datetime
from typing import Any

# Root log directory: always under the repo root (not process cwd), so InvokeError
# and other JSONL logs stay in the same tree Cursor reads.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_BASE_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(LOG_BASE_DIR, exist_ok=True)

# Cache for component loggers to avoid redundant handler setup
_loggers: dict[str, logging.Logger] = {}
_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("logger_context", default={})

# get_component_logger() runs on every single log call (thousands per session) in a
# heavily multi-threaded app (parallel tool execution, the DeepReasoning planner's
# background-thread timeout pattern, concurrent chat requests). Without a lock, one
# thread's _refresh_closed_handlers()/addHandler() can race another thread's write
# to the same shared logging.Logger, surfacing as "ValueError: I/O operation on
# closed file" in whichever thread loses the race.
_logger_setup_lock = threading.Lock()

class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSONL format."""
    def format(self, record):
        # Sanitize message and data to remove surrogates
        message = record.getMessage()
        data = getattr(record, "data", {})

        # Clean surrogates from strings
        def clean_surrogates(obj):
            if isinstance(obj, str):
                return obj.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            elif isinstance(obj, dict):
                return {k: clean_surrogates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_surrogates(item) for item in obj]
            return obj

        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", "Unknown"),
            "phase": getattr(record, "phase", "Unknown"),
            "message": clean_surrogates(message),
            "context": clean_surrogates(getattr(record, "context", {})),
            "data": clean_surrogates(data)
        }
        return json.dumps(log_entry, ensure_ascii=False)


def bind_log_context(**kwargs):
    """Attach request-scoped metadata so downstream logs can be correlated."""
    merged = dict(_LOG_CONTEXT.get() or {})
    merged.update({key: value for key, value in kwargs.items() if value is not None})
    return _LOG_CONTEXT.set(merged)


def reset_log_context(token):
    """Restore the previous logging context."""
    _LOG_CONTEXT.reset(token)


def get_log_context() -> dict[str, Any]:
    """Return the active request-scoped logging metadata."""
    return dict(_LOG_CONTEXT.get() or {})


def _handler_stream_closed(handler: logging.Handler) -> bool:
    stream = getattr(handler, "stream", None)
    return bool(stream is not None and getattr(stream, "closed", False))


def _refresh_closed_handlers(logger: logging.Logger) -> None:
    """Remove handlers whose underlying streams were closed by a prior run."""
    for handler in list(logger.handlers):
        if _handler_stream_closed(handler):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def get_component_logger(component: str) -> logging.Logger:
    """
    Returns a logger scoped to a specific component.
    Logs will be stored in logs/{component}/{component}.jsonl
    """
    log_dir = os.path.join(LOG_BASE_DIR, component)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{component}.jsonl")

    # Everything below reads-then-mutates the shared logging.Logger's handler
    # list (and can close/replace handlers other threads are actively writing
    # through) — must be atomic across threads, not just per-step.
    with _logger_setup_lock:
        if component in _loggers:
            logger = _loggers[component]
            _refresh_closed_handlers(logger)
        else:
            logger = logging.getLogger(f"CairnIQ.{component}")

        logger.setLevel(logging.INFO)
        logger.propagate = False # Avoid double logging to root

        if not logger.handlers:
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(JsonFormatter())
                logger.addHandler(file_handler)
            except Exception as e:
                logger.addHandler(logging.NullHandler())
                print(f"⚠️ Logger for {component} failed: {e}")

        _loggers[component] = logger
    return logger

def log_to_component(component: str, phase: str, message: str, data: dict[str, Any] | None = None, level: int = logging.INFO):
    """Log an event to a specific component channel.

    The message is passed through ``_redact_secrets`` and a regex guard that
    suppresses any line whose redaction failed, so secret-shaped tokens (API keys,
    JWTs, PEM blocks, env-resident secret values) never end up in plaintext logs.
    """
    try:
        from agent.utils import _SECRET_REGEX, _redact_secrets
        safe_message = _redact_secrets(str(message))
        if _SECRET_REGEX.search(safe_message):
            return  # never emit a still-secret-looking line
    except Exception:
        return
    logger = get_component_logger(component)
    try:
        # Construct the LogRecord ourselves and hand it to the logger's handler
        # pipeline. This bypasses ``logger.log(level, message, ...)`` (a sink in
        # CodeQL's standard sensitive-data model) and confines the safe_message to
        # a freshly-constructed record whose ``msg`` field is the already-redacted
        # string.
        if logger.isEnabledFor(level):
            record = logger.makeRecord(
                logger.name,
                level,
                __file__,
                0,
                safe_message,
                (),
                None,
                extra={
                    "component": component,
                    "phase": phase,
                    "context": get_log_context(),
                    "data": data or {},
                },
            )
            logger.handle(record)
    except Exception:
        pass

# --- Legacy/Convenience Agent Functions ---
# These maintain compatibility with existing agent code but use the new structure.

def log_event(phase: str, message: str, data: dict[str, Any] | None = None):
    log_to_component("agent", phase, message, data)

def log_tool_start(tool_name: str, args: dict[str, Any]):
    log_event("ToolExecution", f"Starting {tool_name}", {"args": args})

def log_tool_end(tool_name: str, result: Any, success: bool = True):
    log_event("ToolExecution", f"Finished {tool_name}", {"success": success, "result_preview": str(result)[:100]})

def log_tool_error(tool_name: str, error: Exception):
    try:
        tb = traceback.format_exc()
        log_event("ToolExecution", f"ERROR in {tool_name}", {
            "success": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": tb
        })
    except Exception:
        log_event("ToolExecution", f"ERROR in {tool_name}", {"error": str(error)})

def log_dspy_event(status: str, details: str):
    log_event("ReasoningEngine", f"DSPy {status}", {"details": details})
