"""This module tests methods of :class:`MatrixPairMultiplicationTaskGenerator` in :module:`matrix_multiplication.commands.matrix_multiplication`
"""
import unittest
from unittest.mock import Mock, call

from matrix_multiplication.commands.matrix_multiplication import MatrixPairMultiplicationTaskGenerator


class TestIter(unittest.TestCase):
    """This test case checks if the task builder's __iter__ method for matrix pair multiplication works correctly
    """

    def test_zero_sized_matrices_generate_zero_tasks(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=0)
        matrix2.row_len = Mock(return_value=0)
        task_generator = MatrixPairMultiplicationTaskGenerator(
            matrix1=matrix1, matrix2=matrix2, task_factory=Mock())
        self.assertEqual(matrix1.column_len() * matrix2.row_len(),
                         len([task for task in task_generator.__iter__()]))

    def test_first_matrix_with_zero_rows_called_get_row_zero_times(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=0)
        matrix2.row_len = Mock(return_value=0)
        task_generator = MatrixPairMultiplicationTaskGenerator(
            matrix1=matrix1, matrix2=matrix2, task_factory=Mock())
        tasks = [task for task in task_generator.__iter__()]
        matrix1.get_row.assert_has_calls(
            [call(i) for i in range(0, matrix1.column_len())], any_order=True)

    def test_second_matrix_with_zero_cols_called_get_column_zero_times(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=0)
        matrix2.row_len = Mock(return_value=0)
        task_generator = MatrixPairMultiplicationTaskGenerator(
            matrix1=matrix1, matrix2=matrix2, task_factory=Mock())
        tasks = [task for task in task_generator.__iter__()]
        matrix2.get_column.assert_has_calls(
            [call(i) for i in range(0, matrix2.row_len())], any_order=True)

    def test_matrix1_with_one_row_and_matrix2_with_one_col_generate_single_task(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=1)
        matrix2.row_len = Mock(return_value=1)
        task_generator = MatrixPairMultiplicationTaskGenerator(
            matrix1=matrix1, matrix2=matrix2, task_factory=Mock())
        self.assertEqual(matrix1.column_len() * matrix2.row_len(),
                         len([task for task in task_generator.__iter__()]))

    def test_matrix1_with_ten_rows_and_matrix2_with_ten_cols_generate_hundreed_tasks(self):
        matrix1, matrix2 = Mock(), Mock()
        matrix1.column_len = Mock(return_value=10)
        matrix2.row_len = Mock(return_value=10)
        task_generator = MatrixPairMultiplicationTaskGenerator(
            matrix1=matrix1, matrix2=matrix2, task_factory=Mock())
        self.assertEqual(matrix1.column_len() * matrix2.row_len(),
                         len([task for task in task_generator.__iter__()]))


if __name__ == "__main__":
    unittest.main()
