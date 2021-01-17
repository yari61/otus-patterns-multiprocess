"""This module tests methods of :class:`ValidateMatrixSequenceCommand` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
from __future__ import annotations
import unittest
from unittest.mock import Mock, call

from dependency_injector import containers, providers

from matrix_multiplication.abc.task import Task
from matrix_multiplication.commands.matrix_multiplication import ValidateMatrixSequence


class CommandContainer(containers.DeclarativeContainer):
    validate_matrix_pair = providers.Factory(Task)

    validate_matrix_sequence = providers.Factory(
        ValidateMatrixSequence,
        validate_matrix_pair=validate_matrix_pair
    )


class TestCall(unittest.TestCase):
    def test_task_called_for_each_pair(self):
        validate_matrix_pair = Mock()
        matrices = [Mock() for i in range(10)]
        container = CommandContainer()
        command = container.validate_matrix_sequence(validate_matrix_pair=validate_matrix_pair)
        command(matrices=matrices)
        validate_matrix_pair.assert_has_calls([call(matrix1=matrices[i], matrix2=matrices[i+1]) for i in range(0, len(matrices) - 1)], any_order=True)

if __name__ == "__main__":
    unittest.main()
