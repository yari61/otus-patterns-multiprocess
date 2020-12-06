import multiprocessing
import unittest
from unittest.mock import Mock

import numpy
from dependency_injector import containers, providers

from matrix_multiplication.matrix.adapters import OneDimensionalListMatrixAdapter
from matrix_multiplication.task.processor import MultiprocessTaskProcessor
from matrix_multiplication.commands.matrix_multiplication import\
    MatrixPairMultiplicationCommand,\
    MatrixPairMultiplicationTaskGenerator,\
    CalculateMatrixCellValueCommand
from tests.utils.generate_matrices import generate_valid_matrix_pair


class CommandContainer(containers.DeclarativeContainer):
    matrix_adapter_factory = providers.Factory(
        OneDimensionalListMatrixAdapter
    )
    
    cell_calculation_factory = providers.Factory(
        CalculateMatrixCellValueCommand
    )
    
    task_generator_factory = providers.Factory(
        MatrixPairMultiplicationTaskGenerator,
        task_factory=cell_calculation_factory.provider
    )

    task_processor_factory = providers.Factory(
        MultiprocessTaskProcessor
    )

    pair_multiplication_factory = providers.Factory(
        MatrixPairMultiplicationCommand,
        task_generator_factory=task_generator_factory.provider,
        task_processor_factory=task_processor_factory.provider,
        matrix_adapter_factory=matrix_adapter_factory.provider
    )


class TestMultiprocessMatrixPairMultiplication(unittest.TestCase):
    def test_zero_matrices_multiplication(self):
        container = CommandContainer()
        matrix1, matrix2 = generate_valid_matrix_pair()
        with multiprocessing.Pool() as pool:
            container.task_processor_factory.add_kwargs(pool=pool)
            command = container.pair_multiplication_factory(matrix1=matrix1, matrix2=matrix2)
            result_matrix = command()
            expected_result_matrix = numpy.zeros(shape=(matrix1.column_len(), matrix2.row_len()))
            self.assertTrue(numpy.array_equal(result_matrix._matrix, expected_result_matrix))

if __name__ == "__main__":
    unittest.main()
