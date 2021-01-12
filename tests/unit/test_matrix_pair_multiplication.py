"""This module tests methods of :class:`MatrixPairMultiplicationCommand` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
import unittest
from unittest.mock import Mock, MagicMock

from dependency_injector import providers, containers

from matrix_multiplication.abc.task import TaskGenerator, TaskProcessor
from matrix_multiplication.abc.matrix import ABCMatrix
from matrix_multiplication.commands.matrix_multiplication import MatrixPairMultiplicationCommand


class CommandContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    task_generator_factory = providers.Factory(TaskGenerator)
    task_processor_factory = providers.Factory(TaskProcessor)
    matrix_adapter_factory = providers.Factory(ABCMatrix)

    command_factory = providers.Factory(
        MatrixPairMultiplicationCommand,
        matrix1=config.matrix1,
        matrix2=config.matrix2,
        task_generator_factory=task_generator_factory.provider,
        task_processor_factory=task_processor_factory.provider,
        matrix_adapter_factory=matrix_adapter_factory.provider
    )


class TestCall(unittest.TestCase):
    """This test case checks if the __call__ method of matrix pair multiplication command works correctly
    """

    def test_task_generator_initialized_once_with_matrix1_and_matrix2(self):
        task_generator_factory = MagicMock()
        matrix1, matrix2 = Mock(), Mock()
        container = CommandContainer()
        command = container.command_factory(
            matrix1=matrix1, matrix2=matrix2, task_generator_factory=task_generator_factory, task_processor_factory=Mock(), matrix_adapter_factory=Mock())
        command()
        task_generator_factory.assert_called_once_with(
            matrix1=matrix1, matrix2=matrix2)

    def test_task_processor_initialized_once_with_generated_tasks(self):
        task_processor_factory = Mock()
        matrix1, matrix2 = Mock(), Mock()
        container = CommandContainer(task_generator_factory=MagicMock(
            TaskGenerator), matrix_adapter_factory=Mock(ABCMatrix))
        command = container.command_factory(
            matrix1=matrix1, matrix2=matrix2, task_processor_factory=task_processor_factory)
        command()
        task_processor_factory.assert_called_once_with(
            tasks=[task for task in container.task_generator_factory().__iter__()])

    def test_result_matrix_adapter_initialized_once(self):
        matrix_adapter_factory = Mock()
        matrix1, matrix2 = Mock(), Mock()
        container = CommandContainer(task_generator_factory=MagicMock(
            TaskGenerator), task_processor_factory=MagicMock(TaskProcessor))
        command = container.command_factory(
            matrix1=matrix1, matrix2=matrix2, matrix_adapter_factory=matrix_adapter_factory)
        command()
        matrix_adapter_factory.assert_called_once()

    def test_matrix_returned(self):
        matrix_adapter_factory = Mock()
        matrix1, matrix2 = Mock(), Mock()
        container = CommandContainer(task_generator_factory=MagicMock(
            TaskGenerator), task_processor_factory=MagicMock(TaskProcessor))
        command = container.command_factory(
            matrix1=matrix1, matrix2=matrix2, matrix_adapter_factory=matrix_adapter_factory)
        self.assertEqual(command(), matrix_adapter_factory())


if __name__ == "__main__":
    unittest.main()
