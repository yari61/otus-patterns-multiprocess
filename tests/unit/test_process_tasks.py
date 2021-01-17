"""This module tests methods of :class:`MultiprocessTaskProcessor` in :module:`matrix_multiplication.task.processor`
"""
from __future__ import annotations
import unittest
import unittest.mock as mock
from multiprocessing.pool import Pool

from dependency_injector import containers, providers

from matrix_multiplication.abc.task import Task
from matrix_multiplication.task.processor import MultiprocessTaskProcessor


class Container(containers.DeclarativeContainer):
    processor = providers.Factory(MultiprocessTaskProcessor)


class TestCall(unittest.TestCase):
    def test_each_task_ran_once(self):
        tasks = [mock.Mock(Task) for i in range(10)]
        pool = mock.Mock(Pool)
        container = Container()
        processor = container.processor(pool=pool)
        processor(tasks=tasks)
        pool.apply_async.assert_has_calls([mock.call(task) for task in tasks], any_order=True)


if __name__ == "__main__":
    unittest.main()
