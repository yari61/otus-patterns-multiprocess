import unittest
import random
from unittest.mock import Mock, patch, call

from matrix_multiplication.commands.matrix_multiplication import MatrixPairMultiplicationTaskGenerator
from tests.utils.generate_matrices import generate_valid_matrix_pair


class TestTaskGenerator(unittest.TestCase):
    """This test case checks if the task builder for matrix pair multiplication works correctly
    """
    def test_built_tasks_amount_correct(self):
        matrix1, matrix2 = generate_valid_matrix_pair()
        task_generator = MatrixPairMultiplicationTaskGenerator(matrix1=matrix1, matrix2=matrix2, task_factory=Mock())
        self.assertEqual(matrix1.column_len() * matrix2.row_len(), len([task for task in task_generator.__iter__()]))

    def test_required_methods_called(self):
        matrix1, matrix2 = generate_valid_matrix_pair()
        task_generator = MatrixPairMultiplicationTaskGenerator(matrix1=matrix1, matrix2=matrix2, task_factory=Mock())
        tasks = [task for task in task_generator.__iter__()]
        matrix1.get_row.assert_has_calls([call(i) for i in range(0, matrix1.column_len())], any_order=True)
        matrix2.get_column.assert_has_calls([call(i) for i in range(0, matrix2.row_len())], any_order=True)

if __name__ == "__main__":
    unittest.main()
