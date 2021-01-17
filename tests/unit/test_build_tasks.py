"""This module tests methods of :class:`BuildTasks` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
from __future__ import annotations
import unittest
from unittest.mock import Mock, call

from matrix_multiplication.commands.matrix_multiplication import BuildTasks


class TestCall(unittest.TestCase):
    """This test case checks if the task builder's __iter__ method for matrix pair multiplication works correctly
    """

    def test_zero_sized_matrices_generate_zero_tasks(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=0)
        matrix2.row_len = Mock(return_value=0)
        build_tasks = BuildTasks()
        tasks = build_tasks(matrix1=matrix1, matrix2=matrix2)
        self.assertEqual(matrix1.column_len() * matrix2.row_len(), len(tasks))

    def test_first_matrix_with_zero_rows_called_get_row_zero_times(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=0)
        matrix2.row_len = Mock(return_value=0)
        build_tasks = BuildTasks()
        tasks = build_tasks(matrix1=matrix1, matrix2=matrix2)
        self.assertEqual(matrix1.column_len() * matrix2.row_len(), len(tasks))

    def test_second_matrix_with_zero_cols_called_get_column_zero_times(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=0)
        matrix2.row_len = Mock(return_value=0)
        build_tasks = BuildTasks()
        tasks = build_tasks(matrix1=matrix1, matrix2=matrix2)
        self.assertEqual(matrix1.column_len() * matrix2.row_len(), len(tasks))

    def test_matrix1_with_one_row_and_matrix2_with_one_col_generate_single_task(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=1)
        matrix2.row_len = Mock(return_value=1)
        build_tasks = BuildTasks()
        tasks = build_tasks(matrix1=matrix1, matrix2=matrix2)
        self.assertEqual(matrix1.column_len() * matrix2.row_len(), len(tasks))

    def test_matrix1_with_ten_rows_and_matrix2_with_ten_cols_generate_hundreed_tasks(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=10)
        matrix2.row_len = Mock(return_value=10)
        build_tasks = BuildTasks()
        tasks = build_tasks(matrix1=matrix1, matrix2=matrix2)
        self.assertEqual(matrix1.column_len() * matrix2.row_len(), len(tasks))


if __name__ == "__main__":
    unittest.main()
