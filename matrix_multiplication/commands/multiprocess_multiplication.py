import typing
import collections
import functools
import multiprocessing

import numpy

from matrix_multiplication.interfaces.matrix import IMatrix
from matrix_multiplication.commands.validate_matrices import ValidateMatricesCommand
from matrix_multiplication.commands.calculate_cell_value import CalculateCellValueCommand
from matrix_multiplication.adapters.matrix import NDArrayMatrixAdapter


class MultiprocessMultiplicationMatrixPairTaskBuilder(typing.Iterable):
    __slots__ = ("_pool", "_matrix1", "_matrix2")

    def __init__(self, matrix1: IMatrix, matrix2: IMatrix):
        self._matrix1 = matrix1
        self._matrix2 = matrix2

    def __iter__(self) -> typing.Iterator[typing.Callable]:
        for row_index in range(0, self._matrix1.column_len()):
            for column_index in range(0, self._matrix2.row_len()):
                yield CalculateCellValueCommand(row=self._matrix1.get_row(row_index), column=self._matrix2.get_column(column_index))


class MultiprocessMultiplicationMatrixPairCommand(typing.Callable):
    __slots__ = ("_pool", "_matrix1", "_matrix2")

    def __init__(self, pool: multiprocessing.Pool, matrix1: IMatrix, matrix2: IMatrix):
        self._pool = pool
        self._matrix1 = matrix1
        self._matrix2 = matrix2

    def __call__(self) -> IMatrix:
        """Performs the multiplication of two matrices

        Returns:
            IMatrix: Result matrix
        """

        task_builder = MultiprocessMultiplicationMatrixPairTaskBuilder(self._matrix1, self._matrix2)
        # here tasks are spread between a pool of workers (processes)
        tasks = [self._pool.apply_async(task) for task in task_builder]
        # waiting for tasks completion
        task_results = collections.deque(task.get() for task in tasks)
        # creating result matrix
        result_matrix = numpy.zeros(shape=(self._matrix1.column_len(), self._matrix2.row_len()))
        # filling result matrix
        for row_index in range(0, self._matrix1.column_len()):
            for column_index in range(0, self._matrix2.row_len()):
                result_matrix[row_index][column_index] = task_results.popleft()
        return NDArrayMatrixAdapter(matrix=result_matrix)


class MultiprocessMultiplicationCommand(typing.Callable):
    """This command performs the multiprocess multiplication of matrices sequencially
    """
    __slots__ = ("_matrices", "_pool")

    def __init__(self, pool: multiprocessing.Pool, *matrices: typing.List[IMatrix]) -> None:
        self._matrices = matrices
        self._pool = pool

    def __call__(self) -> IMatrix:
        """Performs the multiplication of matrices sequence

        Raises:
            ValueError: If some matrices pair could not be multiplied

        Returns:
            IMatrix: Result matrix
        """

        if not ValidateMatricesCommand(*self._matrices).__call__():
            raise ValueError("matrices could not be multiplied")
        return functools.reduce(lambda matrix1, matrix2: MultiprocessMultiplicationMatrixPairCommand(self._pool, matrix1, matrix2).__call__(), self._matrices)
