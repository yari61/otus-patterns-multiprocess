"""This module tests methods of :class:`MatrixSequenceMultiplicationCommand` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
import unittest
from unittest.mock import Mock, call

from dependency_injector import containers, providers

from matrix_multiplication.abc.task import Task
from matrix_multiplication.commands.matrix_multiplication import MatrixSequenceMultiplicationCommand


class CommandContainer(containers.DeclarativeContainer):
    pair_multiplication_command = providers.Factory(Task)

    matrix_sequence_multiplication = providers.Factory(
        MatrixSequenceMultiplicationCommand,
        matrix_pair_multiplication_command_factory=pair_multiplication_command.provider
    )


class TestCall(unittest.TestCase):
    def test_task_created_for_each_pair(self):
        pair_multiplication_command = Mock()
        pair_multiplication_command_factory = Mock(
            return_value=pair_multiplication_command)
        matrices = [Mock() for i in range(10)]
        container = CommandContainer()
        command = container.matrix_sequence_multiplication(
            matrices=matrices, matrix_pair_multiplication_command_factory=pair_multiplication_command_factory)
        command()
        pair_multiplication_command_factory.assert_has_calls(
            [call(matrix1=matrices[i] if i == 0 else pair_multiplication_command(), matrix2=matrices[i+1]) for i in range(0, len(matrices) - 1)], any_order=True)

    def test_task_called_for_each_pair(self):
        pair_multiplication_command = Mock()
        pair_multiplication_command_factory = Mock(
            return_value=pair_multiplication_command)
        matrices = [Mock() for i in range(10)]
        container = CommandContainer()
        command = container.matrix_sequence_multiplication(
            matrices=matrices, matrix_pair_multiplication_command_factory=pair_multiplication_command_factory)
        command()
        pair_multiplication_command.assert_has_calls(
            [call() for i in range(0, len(matrices) - 1)], any_order=False)

    def test_pair_mult_equals_to_sequence_mult_if_pair_mult_is_const(self):
        pair_multiplication_command = Mock()
        pair_multiplication_command_factory = Mock(
            return_value=pair_multiplication_command)
        matrices = [Mock() for i in range(10)]
        container = CommandContainer()
        command = container.matrix_sequence_multiplication(
            matrices=matrices, matrix_pair_multiplication_command_factory=pair_multiplication_command_factory)
        self.assertEqual(command(), pair_multiplication_command())


if __name__ == "__main__":
    unittest.main()
