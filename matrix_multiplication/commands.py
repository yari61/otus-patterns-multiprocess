import functools
import typing
import collections
import multiprocessing

import numpy

from .matrix.abc import IMatrix
from .matrix import MatrixAdapterContainer


class CalculateMatrixCellValueCommand(typing.Callable):
    """This command calculates the value of the cell based on the first matrix row and the second matrix column
    """
    __slots__ = ("_row", "_column")

    def __init__(self, row: typing.Iterable[typing.SupportsFloat], column: typing.Iterable[typing.SupportsFloat]) -> None:
        self._row = row
        self._column = column

    def __call__(self) -> typing.SupportsFloat:
        """This command calculates the value of the cell based on the first matrix row and the second matrix column

        Raises:
            ValueError: if lengths of the first matrix row and second matrix column are not equal

        Returns:
            typing.SupportsFloat: result matrix cell value
        """

        # if len(self._row) != len(self._column):
            # raise ValueError(f"row and column lengths are not equal: row {len(self._row)}, column {len(self._column)}")
        cell_value = functools.reduce(lambda previous_sum, value_pair: value_pair[0] * value_pair[1] + previous_sum, zip(self._row, self._column), 0)
        return cell_value


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


class ValidateMatrixSequenceCommand(typing.Callable):
    """This command checks if the sequence of matrices could be multiplied
    """
    __slots__ = ("_matrices",)

    def __init__(self, matrices: typing.List[IMatrix]) -> None:
        self._matrices = matrices

    def __call__(self) -> bool:
        """Checks if first matrix columns count equals to second matrix rows count for each pair of consecutive matrices in matrix sequence

        Returns:
            bool: True if sequence is valid, else False
        """

        for i in range(0, len(self._matrices) - 1):
            validation_command = ValidateMatrixPairCommand(matrix1=self._matrices[i], matrix2=self._matrices[i+1])
            if not validation_command.__call__():
                return False
        return True


class MatrixPairMultiplicationTaskBuilder(typing.Iterable):
    """Builds tasks of calculation of each cell of a result matrix for matrix pair multiplication
    """
    __slots__ = ("_matrix1", "_matrix2")

    def __init__(self, matrix1: IMatrix, matrix2: IMatrix):
        self._matrix1 = matrix1
        self._matrix2 = matrix2

    def __iter__(self) -> CalculateMatrixCellValueCommand:
        """Iterates over cell value calculation tasks in order from the upper left cell to the lower right walking through rows

        Returns:
            CalculateMatrixCellValueCommand: cell value calculation task

        Yields:
            Iterator[CalculateMatrixCellValueCommand]: cell value calculation tasks
        """

        for row_index in range(0, self._matrix1.column_len()):
            for column_index in range(0, self._matrix2.row_len()):
                yield CalculateMatrixCellValueCommand(row=self._matrix1.get_row(row_index), column=self._matrix2.get_column(column_index))


class MultiprocessMatrixPairMultiplicationCommand(typing.Callable):
    """Performs the multiprocess multiplication of two matrices
    """
    __slots__ = ("_pool", "_matrix1", "_matrix2")

    _matrix_adapter_container = MatrixAdapterContainer()

    def __init__(self, pool: multiprocessing.Pool, matrix1: IMatrix, matrix2: IMatrix):
        self._pool = pool
        self._matrix1 = matrix1
        self._matrix2 = matrix2

    def __call__(self) -> IMatrix:
        """Performs the multiprocess multiplication of two matrices

        Returns:
            IMatrix: Result matrix
        """

        task_builder = MatrixPairMultiplicationTaskBuilder(self._matrix1, self._matrix2)
        # here tasks are spread between a pool of workers (processes)
        tasks = [self._pool.apply_async(task) for task in task_builder.__iter__()]
        # waiting for tasks completion
        task_results = collections.deque(task.get() for task in tasks)
        # creating result matrix
        result_matrix = numpy.zeros(shape=(self._matrix1.column_len(), self._matrix2.row_len()))
        # filling result matrix
        for row_index in range(0, self._matrix1.column_len()):
            for column_index in range(0, self._matrix2.row_len()):
                result_matrix[row_index][column_index] = task_results.popleft()
        return self._matrix_adapter_container.matrix_adapter_factory.resolve(result_matrix)


class MultiprocessMatrixSequenceMultiplicationCommand(typing.Callable):
    """Performs the multiprocess multiplication of matrix sequence
    """
    __slots__ = ("_matrices", "_pool")

    def __init__(self, pool: multiprocessing.Pool, matrices: typing.List[IMatrix]) -> None:
        self._matrices = matrices
        self._pool = pool

    def __call__(self) -> IMatrix:
        """Performs the multiprocess multiplication of matrix sequence

        Returns:
            IMatrix: Result matrix
        """

        # firstly multiplying first and second matrices of the sequence
        # then multiplying each result of previous multiplication with the next matrix in the sequence
        return functools.reduce(lambda matrix1, matrix2: MultiprocessMatrixPairMultiplicationCommand(self._pool, matrix1, matrix2).__call__(), self._matrices)
