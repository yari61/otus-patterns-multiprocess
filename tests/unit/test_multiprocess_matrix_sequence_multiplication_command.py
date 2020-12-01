import unittest
import random
from unittest.mock import Mock, patch, call

from matrix_multiplication.commands import MultiprocessMatrixSequenceMultiplicationCommand, MultiprocessMatrixPairMultiplicationCommand, ValidateMatrixSequenceCommand


class TestMatrixPairMultiplication(unittest.TestCase):
    """This test case checks if the matrix pair multiplication command works correctly
    """
    def test_tasks_execution_order(self):
        # creating mock matrices
        matrices = [Mock() for i in range(0, random.randint(5, 10))]
        pool = Mock()
        with patch.object(ValidateMatrixSequenceCommand, "__call__", return_value=True) as mock_validation_method:
            with patch.object(MultiprocessMatrixPairMultiplicationCommand, "__init__", return_value=None) as mock_init_method:
                with patch.object(MultiprocessMatrixPairMultiplicationCommand, "__call__", return_value=None) as mock_call_method:
                    command = MultiprocessMatrixSequenceMultiplicationCommand(pool=pool, matrices=matrices)
                    result_matrix = command()
                    # Checking that matrices were multiplied in the correct order
                    mock_init_method.assert_has_calls([call(pool, matrices[i] if i == 0 else None, matrices[i+1]) for i in range(0, len(matrices) - 1)])
                    # Checking that all multiplication commands were executed
                    mock_call_method.assert_has_calls([call() for i in range(0, len(matrices) - 1)])

if __name__ == "__main__":
    unittest.main()
