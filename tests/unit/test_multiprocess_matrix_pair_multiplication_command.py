import unittest
import random
from unittest.mock import Mock, patch, call

from matrix_multiplication.commands.matrix_multiplication import MultiprocessMatrixPairMultiplicationCommand
from matrix_multiplication.commands.task_iterator import MatrixPairMultiplicationTaskIterator


class TestMatrixPairMultiplication(unittest.TestCase):
    """This test case checks if the matrix pair multiplication command works correctly
    """
    def test_tasks_execution_order(self):
        # creating mock matrices
        matrix1 = Mock()
        matrix1.column_len.return_value = random.randint(5, 10)
        matrix2 = Mock()
        matrix2.row_len.return_value = random.randint(5, 10)
        # creating mock process pool, where each task pushed to pool returns 0
        pool = Mock()
        task = Mock()
        task.get.return_value = 0
        pool.apply_async.return_value = task
        # patching iter method of task builder
        # so it would not require matrices with existing rows and columns
        with patch.object(MatrixPairMultiplicationTaskIterator, "__iter__", return_value=[i for i in range(matrix1.column_len() * matrix2.row_len())]) as mock_task_builder:
            command = MultiprocessMatrixPairMultiplicationCommand(pool=pool, matrix1=matrix1, matrix2=matrix2)
            result_matrix = command()
            # Checking that tasks were executed in the correct order
            pool.apply_async.assert_has_calls([call(i) for i in range(matrix1.column_len() * matrix2.row_len())], any_order=False)

if __name__ == "__main__":
    unittest.main()
