"""This module tests methods of :class:`MatrixPairMultiplicationCommand` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
from __future__ import annotations
import unittest
from unittest.mock import Mock, MagicMock

from dependency_injector import providers, containers

from matrix_multiplication.abc.task import TaskProcessor
from matrix_multiplication.abc.matrix_multiplication import ABCTaskManager
from matrix_multiplication.abc.matrix import LeftMultipliableMatrix, RightMultipliableMatrix
from matrix_multiplication.commands.matrix_multiplication import MultiplyMatrixPair


class CommandContainer(containers.DeclarativeContainer):
    task_processor = providers.Factory(TaskProcessor)
    task_manager = providers.Factory(ABCTaskManager)

    command_factory = providers.Factory(
        MultiplyMatrixPair,
        task_processor=task_processor,
        task_manager=task_manager
    )


class TestCall(unittest.TestCase):
    """This test case checks if the __call__ method of matrix pair multiplication command works correctly
    """

    def test_task_manager_generated_tasks_once_with_matrix1_and_matrix2(self):
        task_manager = Mock(ABCTaskManager)
        matrix1, matrix2 = Mock(LeftMultipliableMatrix), Mock(
            RightMultipliableMatrix)
        container = CommandContainer(
            task_manager=task_manager, task_processor=MagicMock(TaskProcessor))
        command = container.command_factory()
        command(matrix1=matrix1, matrix2=matrix2)
        task_manager.build_tasks.assert_called_once_with(
            matrix1=matrix1, matrix2=matrix2)

    def test_task_processor_called_once_with_generated_tasks(self):
        task_processor = Mock(TaskProcessor)
        task_manager = Mock(ABCTaskManager)
        matrix1, matrix2 = Mock(LeftMultipliableMatrix), Mock(
            RightMultipliableMatrix)
        container = CommandContainer(task_processor=task_processor, task_manager=task_manager)
        command = container.command_factory()
        command(matrix1=matrix1, matrix2=matrix2)
        task_processor.assert_called_once_with(
            tasks=task_manager.build_tasks())

    def test_task_manager_aggregated_tasks_once(self):
        task_processor = Mock(TaskProcessor)
        task_manager = Mock(ABCTaskManager)
        matrix1, matrix2 = Mock(LeftMultipliableMatrix), Mock(
            RightMultipliableMatrix)
        container = CommandContainer(task_processor=task_processor, task_manager=task_manager)
        command = container.command_factory()
        command(matrix1=matrix1, matrix2=matrix2)
        task_manager.handle_results.assert_called_once()

    def test_matrix_returned(self):
        task_processor = Mock(TaskProcessor)
        task_manager = Mock(ABCTaskManager)
        matrix1, matrix2 = Mock(LeftMultipliableMatrix), Mock(
            RightMultipliableMatrix)
        container = CommandContainer(task_processor=task_processor, task_manager=task_manager)
        command = container.command_factory()
        self.assertEqual(
            command(matrix1=matrix1, matrix2=matrix2), task_manager.handle_results())


if __name__ == "__main__":
    unittest.main()
