import typing

from dependency_injector import providers

from matrix_multiplication.matrix.abc import IMatrix


class ValidateMatrixPairCommand(typing.Callable):
    """This command checks if the matrix pair could be multiplied
    """
    __slots__ = ("_matrix1", "_matrix2")

    def __init__(self, matrix1: IMatrix, matrix2: IMatrix) -> None:
        self._matrix1 = matrix1
        self._matrix2 = matrix2

    def __call__(self) -> bool:
        """Checks if first matrix columns count equals to second matrix rows count for each pair of consecutive matrices in matrix sequence

        Returns:
            bool: True if sequence is valid, else False
        """

        if self._matrix1.row_len() != self._matrix2.column_len():
            return False
        return True

# this factory creates ValidateMatrixPairCommand
validate_matrix_pair_command_factory = providers.FactoryAggregate(
    default=providers.Factory(ValidateMatrixPairCommand)
)


class ValidateMatrixSequenceCommand(typing.Callable):
    """This command checks if the sequence of matrices could be multiplied
    """
    __slots__ = ("_matrices",)

    def __init__(self, matrices: typing.Iterable[IMatrix]) -> None:
        self._matrices = matrices

    def __call__(self) -> bool:
        """Checks if first matrix columns count equals to second matrix rows count for each pair of consecutive matrices in matrix sequence

        Returns:
            bool: True if sequence is valid, else False
        """

        for i in range(0, len(self._matrices) - 1):
            validation_command = validate_matrix_pair_command_factory("default", matrix1=self._matrices[i], matrix2=self._matrices[i+1])
            if not validation_command.__call__():
                return False
        return True

# this factory creates ValidateMatrixSequenceCommand
validate_matrix_sequence_command_factory = providers.FactoryAggregate(
    default=providers.Factory(ValidateMatrixSequenceCommand)
)
