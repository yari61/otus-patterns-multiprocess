import typing

from dependency_injector import providers

from .cell_calculation import calculate_matrix_cell_value_command_factory
from matrix_multiplication.matrix.abc import IMatrix


class MatrixPairMultiplicationTaskIterator(typing.Iterable):
    """Builds tasks of calculation of each cell of a result matrix for matrix pair multiplication
    """
    __slots__ = ("_matrix1", "_matrix2")

    def __init__(self, matrix1: IMatrix, matrix2: IMatrix):
        self._matrix1 = matrix1
        self._matrix2 = matrix2

    def __iter__(self) -> typing.Callable:
        """Iterates over cell value calculation tasks in order from the upper left cell to the lower right walking through rows

        Returns:
            typing.Callable: cell value calculation task

        Yields:
            Iterator[typing.Callable]: cell value calculation tasks
        """

        for row_index in range(0, self._matrix1.column_len()):
            for column_index in range(0, self._matrix2.row_len()):
                yield calculate_matrix_cell_value_command_factory("default", row=self._matrix1.get_row(row_index), column=self._matrix2.get_column(column_index))

# this factory creates MatrixPairMultiplicationTaskIterator
matrix_pair_multiplication_task_iterator_factory = providers.FactoryAggregate(
    default=providers.Factory(MatrixPairMultiplicationTaskIterator)
)
