import unittest
from unittest.mock import Mock, call

import numpy
from dependency_injector import containers, providers

from matrix_multiplication.commands.matrix_multiplication import MatrixSequenceMultiplicationCommand
from tests.utils.generate_matrices import generate_valid_matrix_sequence


class CommandContainer(containers.DeclarativeContainer):
    multiplication_sequence_factory = providers.Factory(
        MatrixSequenceMultiplicationCommand
    )


class TestMatricesMultiplication(unittest.TestCase):
    def test_task_created_for_each_pair(self):
        container = CommandContainer()
        multiplication_pair_factory = Mock()
        multiplication_pair_factory.return_value.__call__ = Mock(return_value=None)
        matrices = generate_valid_matrix_sequence()
        command = container.multiplication_sequence_factory(matrices=matrices, matrix_pair_multiplication_command_factory=multiplication_pair_factory)
        command()
        multiplication_pair_factory.assert_has_calls(
            [
                call(matrix1=matrices[i] if i == 0 else None, matrix2=matrices[i+1]) for i in range(0, len(matrices) - 1)
            ],
            any_order=True
        )

    def test_task_called_for_each_pair(self):
        container = CommandContainer()
        multiplication_pair_factory = Mock()
        matrices = generate_valid_matrix_sequence()
        command = container.multiplication_sequence_factory(matrices=matrices, matrix_pair_multiplication_command_factory=multiplication_pair_factory)
        command()
        multiplication_pair_factory.return_value.assert_has_calls(
            [
                call() for i in range(0, len(matrices) - 1)
            ],
            any_order=False
        )

if __name__ == "__main__":
    unittest.main()
