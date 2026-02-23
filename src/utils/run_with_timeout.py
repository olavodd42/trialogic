import concurrent.futures
import time

def run_with_timeout(fn, *args, timeout=180, retries=3, backoff=3, **kwargs):
    last_exc = None
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