import unittest
import random
from unittest.mock import Mock, patch, call

from matrix_multiplication.commands import MatrixPairMultiplicationTaskBuilder


class TestTaskBuilder(unittest.TestCase):
    """This test case checks if the task builder for matrix pair multiplication works correctly
    """
    def test_built_tasks_amount(self):
        matrix1 = Mock()
        matrix1.column_len.return_value = random.randint(5, 10)
        matrix2 = Mock()
        matrix2.row_len.return_value = random.randint(5, 10)
        with patch("matrix_multiplication.commands.CalculateMatrixCellValueCommand") as mock_class:
            task_builder = MatrixPairMultiplicationTaskBuilder(matrix1, matrix2)
            self.assertEqual(matrix1.column_len() * matrix2.row_len(), len([task for task in task_builder]))

    def test_required_methods_called(self):
        matrix1 = Mock()
        matrix1.column_len.return_value = random.randint(5, 10)
        matrix2 = Mock()
        matrix2.row_len.return_value = random.randint(5, 10)
        with patch("matrix_multiplication.commands.CalculateMatrixCellValueCommand") as mock_class:
            task_builder = MatrixPairMultiplicationTaskBuilder(matrix1, matrix2)
            tasks = [task for task in task_builder]
            matrix1.get_row.assert_has_calls([call(i) for i in range(0, matrix1.column_len())], any_order=True)
            matrix2.get_column.assert_has_calls([call(i) for i in range(0, matrix2.row_len())], any_order=True)

if __name__ == "__main__":
    unittest.main()
