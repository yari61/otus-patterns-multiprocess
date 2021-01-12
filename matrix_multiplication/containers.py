from dependency_injector import containers, providers

from matrix_multiplication.abc import ABCMatrix
from matrix_multiplication.task import MultiprocessTaskProcessor
from matrix_multiplication.matrix.adapters import OneDimensionalListMatrixAdapter
from matrix_multiplication.commands.matrix_multiplication import (CalculateMatrixCellValueCommand, MatrixPairMultiplicationTaskGenerator,
                                                                  MatrixPairMultiplicationCommand, MatrixSequenceMultiplicationCommand, ValidateMatrixPairCommand, ValidateMatrixSequence)


class MatrixMultiplicationCommandsContainer(containers.DeclarativeContainer):
    matrix_adapter_factory = providers.Factory(OneDimensionalListMatrixAdapter)

    cell_calculation_factory = providers.Factory(
        CalculateMatrixCellValueCommand)

    task_generator_factory = providers.Factory(
        MatrixPairMultiplicationTaskGenerator, task_factory=cell_calculation_factory.provider)

    task_processor_factory = providers.Factory(MultiprocessTaskProcessor)

    pair_multiplication_factory = providers.Factory(MatrixPairMultiplicationCommand, task_generator_factory=task_generator_factory.provider,
                                                    task_processor_factory=task_processor_factory.provider, matrix_adapter_factory=matrix_adapter_factory.provider)

    sequence_multiplication_factory = providers.Factory(
        MatrixSequenceMultiplicationCommand, matrix_pair_multiplication_command_factory=pair_multiplication_factory.provider)

    validate_matrix_pair = providers.Factory(ValidateMatrixPairCommand)

    validate_sequence_factory = providers.Factory(
        ValidateMatrixSequence, validate_matrix_pair_factory=validate_matrix_pair.provider)
