import multiprocessing
import unittest
from unittest.mock import Mock

import numpy

from matrix_multiplication.commands import MultiprocessMatrixPairMultiplicationCommand

from tests.utils.generate_matrices import generate_valid_matrix_pair, generate_invalid_matrix_pair


class TestMultiprocessMatrixPairMultiplication(unittest.TestCase):
    def test_zero_matrices_multiplication(self):
        matrix1, matrix2 = generate_valid_matrix_pair()
        with multiprocessing.Pool() as pool:
            command = MultiprocessMatrixPairMultiplicationCommand(pool=pool, matrix1=matrix1, matrix2=matrix2)
            result_matrix = command()
            expected_result_matrix = numpy.zeros(shape=(matrix1.column_len(), matrix2.row_len()))
            self.assertTrue(numpy.array_equal(result_matrix._matrix, expected_result_matrix))

    def test_invalid_zero_matrices_multiplication(self):
        matrix1, matrix2 = generate_invalid_matrix_pair()
        with multiprocessing.Pool() as pool:
            command = MultiprocessMatrixPairMultiplicationCommand(pool=pool, matrix1=matrix1, matrix2=matrix2)
            with self.assertRaises(ValueError) as context:
                result_matrix = command()

if __name__ == "__main__":
    unittest.main()
