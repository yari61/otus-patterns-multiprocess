"""This module tests methods of :class:`ValidateMatrixPair` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
from __future__ import annotations
import unittest
from unittest.mock import Mock, call

from dependency_injector import containers, providers

from matrix_multiplication.abc.task import Task
from matrix_multiplication.abc.matrix import ABCMatrix
from matrix_multiplication.commands.matrix_multiplication import ValidateMatrixPair


class CommandContainer(containers.DeclarativeContainer):
    validate_matrix_pair = providers.Factory(
        ValidateMatrixPair
    )


class TestCall(unittest.TestCase):
    def test_matrix1_cols_eq_matrix2_rows_returns_true(self):
        matrix1, matrix2 = Mock(ABCMatrix), Mock(ABCMatrix)
        matrix1.row_len = Mock(return_value=0)
        matrix2.column_len = Mock(return_value=0)
        container = CommandContainer()
        command = container.validate_matrix_pair()
        self.assertTrue(command(matrix1=matrix1, matrix2=matrix2))

    def test_matrix1_cols_ne_matrix2_rows_returns_false(self):
        matrix1, matrix2 = Mock(ABCMatrix), Mock(ABCMatrix)
        matrix1.row_len = Mock(return_value=0)
        matrix2.column_len = Mock(return_value=1)
        container = CommandContainer()
        command = container.validate_matrix_pair()
        self.assertFalse(command(matrix1=matrix1, matrix2=matrix2))

if __name__ == "__main__":
    unittest.main()
