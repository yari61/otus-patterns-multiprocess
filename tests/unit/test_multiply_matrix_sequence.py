"""This module tests methods of :class:`MultiplyMatrixSequence` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
from __future__ import annotations
import unittest
from unittest.mock import Mock, call

from dependency_injector import containers, providers

from matrix_multiplication.abc.task import Task
from matrix_multiplication.commands.matrix_multiplication import MultiplyMatrixSequence


class CommandContainer(containers.DeclarativeContainer):
    multiply_matrix_pair = providers.Factory(Task)

    matrix_sequence_multiplication = providers.Factory(
        MultiplyMatrixSequence,
        multiply_matrix_pair=multiply_matrix_pair
    )


class TestCall(unittest.TestCase):
    def test_task_called_for_each_pair(self):
        multiply_matrix_pair = Mock()
        matrices = [Mock() for i in range(10)]
        container = CommandContainer()
        command = container.matrix_sequence_multiplication(multiply_matrix_pair=multiply_matrix_pair)
        command(matrices)
        multiply_matrix_pair.assert_has_calls(
            [call(matrix1=matrices[i] if i==0 else multiply_matrix_pair(), matrix2=matrices[i+1]) for i in range(0, len(matrices) - 1)], any_order=False)

    def test_pair_mult_equals_to_sequence_mult_if_pair_mult_is_const(self):
        multiply_matrix_pair = Mock()
        matrices = [Mock() for i in range(10)]
        container = CommandContainer()
        command = container.matrix_sequence_multiplication(multiply_matrix_pair=multiply_matrix_pair)
        self.assertEqual(command(matrices), multiply_matrix_pair())


if __name__ == "__main__":
    unittest.main()
