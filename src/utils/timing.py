import contextlib
import time
from typing import Generator


@contextlib.contextmanager
def timed_section(label: str) -> Generator[None, None, None]:
    """Context manager to time a section of code.

    Usage:

    ```
    with utils.timed_section("my section"):
        # do stuff
    ```
    """

    t1 = time.time()

    yield

    t2 = time.time()
    print(f"Time to {label}: {t2 - t1:.2f}s")
