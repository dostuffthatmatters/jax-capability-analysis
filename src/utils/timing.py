import contextlib
import time
from typing import Generator


@contextlib.contextmanager
def timed_section(label: str, indent: int = 0) -> Generator[None, None, None]:
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
    print((" " * indent) + f"Time to {label}: {t2 - t1:.6f}s")
