import time
from collections.abc import Iterable
from dataclasses import dataclass

import psutil


@dataclass
class PerformanceMetrics:
    """
    Dataclass to store performance metrics of a program execution.
    """

    # Execution wall time in seconds (float)
    time_wall: float


def run_program_and_measure(program_args: Iterable[str]) -> PerformanceMetrics:
    start_time = time.time()
    process = psutil.Popen(program_args)

    # Wait for the program to finish
    process.wait()

    end_time = time.time()
    elapsed_time = end_time - start_time  # wall time in seconds
    return PerformanceMetrics(time_wall=elapsed_time)
