"""Retry-with-timeout helper for unreliable LLM calls."""

import concurrent.futures
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def run_with_timeout(
    fn: Callable[..., T],
    *args: Any,
    timeout: int = 180,
    retries: int = 3,
    backoff: int = 3,
    **kwargs: Any,
) -> T:
    """Execute *fn* with a per-attempt timeout and exponential back-off retries.

    Args:
        fn: Callable to execute.
        *args: Positional arguments forwarded to *fn*.
        timeout: Maximum seconds to wait per attempt.
        retries: Number of extra attempts after the first failure.
        backoff: Base seconds for the linear back-off sleep.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*.

    Raises:
        TimeoutError: If every attempt exceeds *timeout*.
        Exception: The last exception raised by *fn* if all attempts fail.
    """
    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                last_exc = TimeoutError(f"Timeout after {timeout}s")
            except Exception as e:
                last_exc = e
        time.sleep(backoff * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    else:
        raise RuntimeError("run_with_timeout failed without capturing an exception.")