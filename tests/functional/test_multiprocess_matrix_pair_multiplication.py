import multiprocessing
import unittest
from unittest.mock import Mock
from functools import reduce

import numpy
from dependency_injector import containers, providers

from matrix_multiplication import multiprocess_matrices_multiplication
from matrix_multiplication.matrix.adapters import to_ndarray
from tests.functional.utils.container import MatrixContainer
from tests.functional.utils.matrix import RandomMatrixFactory, ZeroMatrixFactory, MatrixSequenceFactory, ValidShapeSequenceFactory, InvalidShapeSequenceFactory


class TestMultiprocessMatrixSequenceMultiplication(unittest.TestCase):
    def test_zero_matrices_multiplication(self):
        # generating random matrix pair
        matrix_container = MatrixContainer()
        matrix_container.matrix_factory.override(
            providers.Factory(ZeroMatrixFactory))
        matrix_container.matrix_sequence_factory.override(providers.Factory(
            MatrixSequenceFactory, matrix_factory=matrix_container.matrix_factory))
        matrix_container.shape_sequence_factory.override(
            providers.Factory(ValidShapeSequenceFactory))
        generate_matrix_sequence = matrix_container.generate_matrix_sequence()
        matrices = generate_matrix_sequence(length=2)
        # calculating expected result
        expected_result_matrix = numpy.zeros(
            shape=(matrices[0].column_len(), matrices[-1].row_len()))
        # executing command under test
        with multiprocessing.Pool() as pool:
            result_matrix = multiprocess_matrices_multiplication(
                pool=pool, matrices=matrices)
            self.assertTrue(numpy.array_equal(
                to_ndarray(matrix=result_matrix), expected_result_matrix))

    def test_random_matrices_multiplication(self):
        # generating random matrix pair
        matrix_container = MatrixContainer()
        matrix_container.matrix_factory.override(
            providers.Factory(RandomMatrixFactory))
        matrix_container.matrix_sequence_factory.override(providers.Factory(
            MatrixSequenceFactory, matrix_factory=matrix_container.matrix_factory))
        matrix_container.shape_sequence_factory.override(
            providers.Factory(ValidShapeSequenceFactory))
        generate_matrix_sequence = matrix_container.generate_matrix_sequence()
        matrices = generate_matrix_sequence(length=2)
        # calculating expected result
        expected_result_matrix = reduce(lambda m1, m2: numpy.dot(
            m1, to_ndarray(m2)), matrices[1:], to_ndarray(matrices[0]))
        # executing command under test
        with multiprocessing.Pool() as pool:
            result_matrix = multiprocess_matrices_multiplication(
                pool=pool, matrices=matrices)
            numpy.testing.assert_array_almost_equal(
                to_ndarray(matrix=result_matrix), expected_result_matrix)

    def test_invalid_sequence_ten_matrices_multiplication(self):
        # generating random matrix pair
        matrix_container = MatrixContainer()
        matrix_container.matrix_factory.override(
            providers.Factory(ZeroMatrixFactory))
        matrix_container.matrix_sequence_factory.override(providers.Factory(
            MatrixSequenceFactory, matrix_factory=matrix_container.matrix_factory))
        matrix_container.shape_sequence_factory.override(
            providers.Factory(InvalidShapeSequenceFactory, error_count_range=(1, 1)))
        generate_matrix_sequence = matrix_container.generate_matrix_sequence()
        matrices = generate_matrix_sequence(length=2)
        # executing command under test
        with self.assertRaises(ValueError):
            with multiprocessing.Pool() as pool:
                result_matrix = multiprocess_matrices_multiplication(
                    pool=pool, matrices=matrices)


if __name__ == "__main__":
    unittest.main()
