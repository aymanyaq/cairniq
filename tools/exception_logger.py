"""
Exception logging decorator for tools.
Automatically logs all exceptions with context.
"""
import functools
import logging

from agent.logger import log_to_component


def log_exceptions(tool_name=None):
    """
    Decorator that automatically logs exceptions from tool functions.

    Usage:
        @log_exceptions("market_data")
        def get_stock_data(symbol):
            ...

    Or auto-detect from function name:
        @log_exceptions()
        def get_stock_data(symbol):
            ...
    """
    def decorator(func):
        nonlocal tool_name
        if tool_name is None:
            tool_name = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Build context from function arguments
                context = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "function": func.__name__,
                }

                # Add first positional arg if it looks like a symbol/ticker
                if args and isinstance(args[0], str) and len(args[0]) <= 10:
                    context["symbol"] = args[0]

                # Add relevant kwargs
                for key in ['symbol', 'ticker', 'query', 'sector', 'strategy']:
                    if key in kwargs:
                        context[key] = str(kwargs[key])[:100]

                log_to_component(
                    "tools",
                    tool_name,
                    f"Exception in {func.__name__}",
                    context,
                    level=logging.ERROR
                )

                # Re-raise the exception so tool behavior doesn't change
                raise

        return wrapper
    return decorator


def log_and_return_error(tool_name=None, default_return=None):
    """
    Decorator that logs exceptions and returns a default value instead of raising.
    Useful for tools that should gracefully degrade.

    Usage:
        @log_and_return_error("market_data", default_return={})
        def get_stock_data(symbol):
            ...
    """
    def decorator(func):
        nonlocal tool_name
        if tool_name is None:
            tool_name = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Build context
                context = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "function": func.__name__,
                }

                if args and isinstance(args[0], str) and len(args[0]) <= 10:
                    context["symbol"] = args[0]

                for key in ['symbol', 'ticker', 'query', 'sector', 'strategy']:
                    if key in kwargs:
                        context[key] = str(kwargs[key])[:100]

                log_to_component(
                    "tools",
                    tool_name,
                    f"Exception in {func.__name__} - returning default value",
                    context,
                    level=logging.ERROR
                )

                # Return default instead of raising
                return default_return

        return wrapper
    return decorator
