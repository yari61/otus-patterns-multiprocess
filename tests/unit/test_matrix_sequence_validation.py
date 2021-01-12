"""This module tests methods of :class:`ValidateMatrixSequenceCommand` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
import unittest
from unittest.mock import Mock, call

from dependency_injector import containers, providers

from matrix_multiplication.abc.task import Task
from matrix_multiplication.commands.matrix_multiplication import ValidateMatrixSequenceCommand


class CommandContainer(containers.DeclarativeContainer):
    validate_pair = providers.Factory(Task)

    validate_sequence_factory = providers.Factory(
        ValidateMatrixSequenceCommand,
        validate_matrix_pair_command_factory=validate_pair.provider
    )


class TestCall(unittest.TestCase):
    def test_task_created_for_each_pair(self):
        validate_pair_factory = Mock()
        matrices = [Mock() for i in range(10)]
        container = CommandContainer()
        command = container.validate_sequence_factory(matrices=matrices, validate_matrix_pair_command_factory=validate_pair_factory)
        command()
        validate_pair_factory.assert_has_calls([call(matrix1=matrices[i], matrix2=matrices[i+1]) for i in range(0, len(matrices) - 1)], any_order=True)

    def test_task_called_for_each_pair(self):
        validate_pair = Mock(Task)
        matrices = [Mock() for i in range(10)]
        container = CommandContainer(validate_pair=validate_pair)
        command = container.validate_sequence_factory(matrices=matrices)
        command()
        validate_pair.assert_has_calls([call() for i in range(0, len(matrices) - 1)], any_order=False)

if __name__ == "__main__":
    unittest.main()
