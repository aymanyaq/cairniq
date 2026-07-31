"""Regression: get_component_logger() mutates a shared logging.Logger's handler
list on every log call. Production hit intermittent "ValueError: I/O operation
on closed file" from concurrent threads racing through the unlocked
check-then-mutate path (duplicate FileHandlers on the same component, or one
thread closing a handler while another is mid-emit). A module-level lock now
makes that path atomic across threads.
"""
import threading

from agent.logger import _loggers, get_component_logger, log_event


def test_concurrent_first_access_creates_exactly_one_handler():
    """Many threads racing to initialize the SAME never-before-seen component
    logger must not each create their own FileHandler on the same file."""
    component = "test_concurrency_probe"
    _loggers.pop(component, None)

    barrier = threading.Barrier(16)
    errors = []

    def worker():
        barrier.wait()
        try:
            get_component_logger(component)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    logger = _loggers[component]
    assert len(logger.handlers) == 1


def test_concurrent_logging_does_not_raise():
    """Many threads logging to the same component concurrently must not hit
    'I/O operation on closed file' or any other exception."""
    errors = []
    barrier = threading.Barrier(16)

    def worker(i):
        barrier.wait()
        try:
            for _ in range(20):
                log_event("ConcurrencyProbe", "stress test message", {"i": i})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors
