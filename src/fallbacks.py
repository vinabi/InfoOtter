from time import sleep
from typing import Callable, Any
import random

def with_retries(tool_fn: Callable[..., Any], *, attempts: int = 3, base_sleep: float = 0.5):
    def wrapper(*args, **kwargs):
        last_exc = None
        for i in range(attempts):
            try:
                return tool_fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                sleep(base_sleep * (2 ** i) + random.random() * 0.2)
        raise last_exc
    return wrapper

def circuit_broken(failure_count: int, limit: int) -> bool:
    return failure_count >= limit
