import typing
import multiprocessing

from matrix_multiplication.abc.task import TaskProcessor, Task


class MultiprocessTaskProcessor(TaskProcessor):
    __slots__ = ("_pool", "_tasks")

    def __init__(self, pool: multiprocessing.Pool, tasks: typing.Iterable[Task]):
        self._pool = pool
        self._tasks = tasks

    def __call__(self) -> typing.Iterable[object]:
        # here tasks are spread between a pool of workers (processes)
        tasks = [self._pool.apply_async(task) for task in self._tasks]
        # waiting for tasks completion
        results = [task.get() for task in tasks]
        return results
