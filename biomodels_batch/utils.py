"""Utility functions for benchmark suite."""

import functools
import logging
import signal
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

# Set up logging
logger = logging.getLogger(__name__)

# signal.alarm/SIGALRM are Unix-only. Where they are unavailable (e.g. Windows)
# we fall back to a worker-thread timeout. Signals give a hard interrupt of even
# CPU-bound code; the thread fallback cannot kill a runaway worker, so it lets the
# worker keep running as a daemon while returning control (and a TimeoutError) to
# the caller. Both paths preserve in-place mutation of shared arguments made
# before the timeout, since the worker shares the caller's memory.
_HAS_SIGALRM = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")


class TimeoutError(Exception):
    """Raised when a function times out."""

    pass


def _run_with_thread_timeout(
    func: Callable,
    args: tuple,
    kwargs: dict,
    seconds: int,
    message: str,
) -> Any:
    """Run ``func`` in a daemon thread and enforce a wall-clock timeout.

    Used on platforms without ``signal.alarm``. Raises :class:`TimeoutError` if
    the worker does not finish within ``seconds``; otherwise returns the worker's
    result or re-raises the exception it raised.
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def target() -> None:
        try:
            result.append(func(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - propagated to the caller
            error.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(message)
    if error:
        raise error[0]
    return result[0]


def timeout(seconds: int):
    """
    Decorator to add timeout to a function.

    Usage:
        @timeout(30)
        def slow_function():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"Function {func.__name__} timed out after {seconds}s"

            if not _HAS_SIGALRM:
                return _run_with_thread_timeout(func, args, kwargs, seconds, message)

            def handler(signum, frame):
                raise TimeoutError(message)

            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the old handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper

    return decorator


@contextmanager
def timeout_context(seconds: int):
    """
    Context manager for timeout protection.

    Usage:
        with timeout_context(30):
            slow_operation()

    On platforms without ``signal.alarm`` the protected block cannot be
    interrupted mid-execution; instead the elapsed time is checked on exit and
    :class:`TimeoutError` is raised if it exceeded ``seconds``. Use
    :func:`safe_execute` for a hard timeout that also works on Windows.
    """
    message = f"Operation timed out after {seconds}s"

    if not _HAS_SIGALRM:
        start = time.monotonic()
        yield
        if time.monotonic() - start > seconds:
            raise TimeoutError(message)
        return

    def handler(signum, frame):
        raise TimeoutError(message)

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_execute(
    func: Callable, *args, timeout_sec: int | None = None, default: Any = None, **kwargs
) -> tuple[bool, Any, str | None]:
    """
    Safely execute a function with optional timeout and error handling.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        timeout_sec: Optional timeout in seconds
        default: Default value to return on error
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (success: bool, result: Any, error_msg: Optional[str])
    """
    try:
        if timeout_sec:
            if _HAS_SIGALRM:
                with timeout_context(timeout_sec):
                    result = func(*args, **kwargs)
            else:
                result = _run_with_thread_timeout(
                    func,
                    args,
                    kwargs,
                    timeout_sec,
                    f"Operation timed out after {timeout_sec}s",
                )
        else:
            result = func(*args, **kwargs)
        return True, result, None
    except TimeoutError as e:
        return False, default, f"Timeout: {str(e)}"
    except Exception as e:
        return False, default, f"Error: {type(e).__name__}: {str(e)}"


def count_species(antimony_text: str) -> int:
    """Count the number of species in Antimony text."""
    import re

    # Species are defined with initializations like: X = 1.0
    # or appear on LHS of reactions
    species = set()

    # Find initializations (excluding parameters in reactions)
    for line in antimony_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue

        # Match: var = value (but not in reactions)
        if "=" in line and ";" not in line and "->" not in line:
            match = re.match(r"^([A-Za-z_]\w*)\s*=", line)
            if match:
                var_name = match.group(1)
                # Exclude common parameter-like names
                if not any(prefix in var_name.lower() for prefix in ["k", "rate", "param"]):
                    species.add(var_name)

        # Find species in reactions (LHS and RHS)
        if "->" in line:
            # Extract reaction equation part before ';'
            rxn_part = line.split(";")[0] if ";" in line else line
            # Remove reaction labels (label:)
            if ":" in rxn_part:
                rxn_part = rxn_part.split(":", 1)[1]

            # Extract species from both sides of arrow
            for side in rxn_part.split("->"):
                # Split by + and extract species names
                for term in side.split("+"):
                    term = term.strip()
                    # Remove stoichiometry numbers
                    match = re.search(r"([A-Za-z_]\w*)", term)
                    if match:
                        species.add(match.group(1))

    return len(species)


def count_reactions(antimony_text: str) -> int:
    """Count the number of reactions in Antimony text."""
    count = 0
    for line in antimony_text.split("\n"):
        if "->" in line and not line.strip().startswith("//"):
            count += 1
    return count


def count_parameters(antimony_text: str) -> int:
    """Count the number of parameters in Antimony text."""
    import re

    params = set()

    for line in antimony_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        # Match parameter definitions: name = numeric_value
        if "=" in line and "->" not in line:
            match = re.match(r"^([A-Za-z_]\w*)\s*=\s*([0-9.eE+-]+)", line)
            if match:
                params.add(match.group(1))

    return len(params)


def detect_features(antimony_text: str) -> dict[str, bool]:
    """
    Detect various features in Antimony text.

    Returns:
        Dictionary with feature flags
    """
    lower = antimony_text.lower()

    # Count occurrences
    piecewise_count = lower.count("piecewise")

    return {
        "events": " at " in lower or "event" in lower,
        "delays": "delay(" in lower,
        "algebraic_rules": ":=" in antimony_text and "var " in lower,
        "piecewise": "piecewise" in lower or "?:" in antimony_text,
        "piecewise_heavy": piecewise_count > 2,
        "time_dependent": any(tok in lower for tok in [" time ", " t)"]),
        "sin": " sin(" in lower,
        "cos": " cos(" in lower,
        "tan": " tan(" in lower,
        "tanh": " tanh(" in lower,
        "sign_changing_trig": any(f in lower for f in [" sin(", " cos(", " tanh("]),
        "exp": " exp(" in lower,
        "log": " log(" in lower,
        "sqrt": " sqrt(" in lower,
    }


def load_fetch_history() -> dict:
    """Load fetch history from JSON file."""
    import json
    from pathlib import Path

    # Import config here to avoid circular imports
    try:
        import config

        history_path = Path(config.FETCH_HISTORY)
    except Exception:
        history_path = Path("data/fetch_history.json")

    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return {"fetch_sessions": [], "total_unique_models": 0, "all_fetched_ids": []}


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
