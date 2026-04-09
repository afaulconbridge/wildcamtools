import functools
import time
from collections.abc import Callable
from types import TracebackType
from typing import ParamSpec, Self, TypeVar

# used for typing the timer as a decorator
P = ParamSpec("P")
R = TypeVar("R")


class Timer:
    _start: float = 0.0
    elapsed: float = 0.0  # seconds
    intervals: int = 0

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.elapsed += time.perf_counter() - self._start
        self.intervals += 1

    def __call__(
        self,
        func: Callable[P, R],
    ) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with self:
                return func(*args, **kwargs)

        return wrapper

    def reset(self) -> float:
        old = self.elapsed
        self.elapsed = 0.0
        return old

    @property
    def per_second(self) -> float:
        return self.intervals / self.elapsed

    @property
    def per_interval(self) -> float:
        return self.elapsed / self.intervals
