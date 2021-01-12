from dependency_injector import containers, providers

from matrix_multiplication.abc import ABCMatrix
from matrix_multiplication.task import MultiprocessTaskProcessor
from matrix_multiplication.matrix.adapters import OneDimensionalListMatrixAdapter
from matrix_multiplication.commands.matrix_multiplication import (CalculateMatrixCellValueCommand, MatrixPairMultiplicationTaskGenerator,
                                                                  MultiplyMatrixPair, MultiplyMatrixSequence, ValidateMatrixPair, ValidateMatrixSequence)


class MatrixMultiplicationCommandsContainer(containers.DeclarativeContainer):
    matrix_adapter_factory = providers.Factory(OneDimensionalListMatrixAdapter)

    cell_calculation_factory = providers.Factory(
        CalculateMatrixCellValueCommand)

    task_generator_factory = providers.Factory(
        MatrixPairMultiplicationTaskGenerator, task_factory=cell_calculation_factory.provider)

    task_processor_factory = providers.Factory(MultiprocessTaskProcessor)

    multiply_matrix_pair = providers.Factory(MultiplyMatrixPair, task_generator_factory=task_generator_factory.provider,
                                                    task_processor_factory=task_processor_factory.provider, matrix_adapter_factory=matrix_adapter_factory.provider)

    multiply_matrix_sequence = providers.Factory(
        MultiplyMatrixSequence, multiply_matrix_pair=multiply_matrix_pair)

    validate_matrix_pair = providers.Factory(ValidateMatrixPair)

    validate_sequence_factory = providers.Factory(
        ValidateMatrixSequence, validate_matrix_pair=validate_matrix_pair)
