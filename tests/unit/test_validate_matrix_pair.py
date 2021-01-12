"""This module tests methods of :class:`ValidateMatrixPairCommand` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
import unittest
from unittest.mock import Mock, call

from dependency_injector import containers, providers

from matrix_multiplication.abc.task import Task
from matrix_multiplication.abc.matrix import ABCMatrix
from matrix_multiplication.commands.matrix_multiplication import ValidateMatrixPairCommand


class CommandContainer(containers.DeclarativeContainer):
    validate_pair = providers.Factory(
        ValidateMatrixPairCommand
    )


class TestCall(unittest.TestCase):
    def test_matrix1_cols_eq_matrix2_rows_returns_true(self):
        matrix1, matrix2 = Mock(ABCMatrix), Mock(ABCMatrix)
        matrix1.row_len = Mock(return_value=0)
        matrix2.column_len = Mock(return_value=0)
        container = CommandContainer()
        command = container.validate_pair(matrix1=matrix1, matrix2=matrix2)
        self.assertTrue(command())

    def test_matrix1_cols_ne_matrix2_rows_returns_false(self):
        matrix1, matrix2 = Mock(ABCMatrix), Mock(ABCMatrix)
        matrix1.row_len = Mock(return_value=0)
        matrix2.column_len = Mock(return_value=1)
        container = CommandContainer()
        command = container.validate_pair(matrix1=matrix1, matrix2=matrix2)
        self.assertFalse(command())

if __name__ == "__main__":
    unittest.main()
