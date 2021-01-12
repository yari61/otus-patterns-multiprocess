"""This module tests multiprocess multiplication of matrix pair
"""
import multiprocessing
import unittest
from unittest.mock import Mock

import numpy
from dependency_injector import containers, providers

from matrix_multiplication.matrix.adapters import OneDimensionalListMatrixAdapter
from matrix_multiplication.task.processor import MultiprocessTaskProcessor
from matrix_multiplication.commands.matrix_multiplication import (
    MultiplyMatrixPair, MatrixPairMultiplicationTaskGenerator, CalculateMatrixCellValueCommand)
from tests.functional.utils.generate_matrices import generate_valid_matrix_pair


class CommandContainer(containers.DeclarativeContainer):
    matrix_adapter = providers.Factory(OneDimensionalListMatrixAdapter)
    cell_calculation = providers.Factory(
        CalculateMatrixCellValueCommand)
    task_generator = providers.Factory(
        MatrixPairMultiplicationTaskGenerator, task_factory=cell_calculation.provider)
    task_processor = providers.Factory(MultiprocessTaskProcessor)
    pair_multiplication = providers.Factory(MultiplyMatrixPair, task_generator_factory=task_generator.provider,
                                                    task_processor_factory=task_processor.provider, matrix_adapter_factory=matrix_adapter.provider)


class TestMultiprocessMatrixPairMultiplication(unittest.TestCase):
    def test_zero_matrix_pair_multiplication(self):
        container = CommandContainer()
        matrix1, matrix2 = generate_valid_matrix_pair()
        with multiprocessing.Pool() as pool:
            container.task_processor.add_kwargs(pool=pool)
            command = container.pair_multiplication()
            result_matrix = command(matrix1=matrix1, matrix2=matrix2)
            expected_result_matrix = numpy.zeros(
                shape=(matrix1.column_len(), matrix2.row_len()))
            self.assertTrue(numpy.array_equal(
                result_matrix._matrix, expected_result_matrix))


if __name__ == "__main__":
    unittest.main()
