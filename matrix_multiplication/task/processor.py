"""Module with concrete implementation of task processor
"""
from __future__ import annotations
import typing
import multiprocessing

from matrix_multiplication.abc.task import TaskProcessor, Task


class MultiprocessTaskProcessor(TaskProcessor):
    __slots__ = ("_pool",)

    def __init__(self, pool: multiprocessing.Pool):
        self._pool = pool

    def __call__(self, tasks: typing.Iterable[Task]) -> typing.Iterable[object]:
        # here tasks are spread between a pool of workers (processes)
        tasks = [self._pool.apply_async(task) for task in tasks]
        # waiting for tasks completion
        results = [task.get() for task in tasks]
        return results
