import functools
import typing
import collections
import multiprocessing

import numpy
from dependency_injector import providers

from .task_iterator import matrix_pair_multiplication_task_iterator_factory
from matrix_multiplication.matrix import MatrixAdapterContainer
from matrix_multiplication.matrix.abc import IMatrix


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

        task_iterator = matrix_pair_multiplication_task_iterator_factory("default", self._matrix1, self._matrix2)
        # here tasks are spread between a pool of workers (processes)
        tasks = [self._pool.apply_async(task) for task in task_iterator.__iter__()]
        # waiting for tasks completion
        task_results = collections.deque(task.get() for task in tasks)
        shape = (self._matrix1.column_len(), self._matrix2.row_len())
        return self._matrix_adapter_container.matrix_adapter_factory.resolve(task_results, shape=shape)

# this factory creates MultiprocessMatrixPairMultiplicationCommand
matrix_pair_multiplication_command_factory = providers.FactoryAggregate(
    multiprocess=providers.Factory(MultiprocessMatrixPairMultiplicationCommand)
)


class MultiprocessMatrixSequenceMultiplicationCommand(typing.Callable):
    """Performs the multiprocess multiplication of matrix sequence
    """
    __slots__ = ("_matrices", "_pool")

    def __init__(self, pool: multiprocessing.Pool, matrices: typing.Iterable[IMatrix]) -> None:
        self._matrices = matrices
        self._pool = pool

    def __call__(self) -> IMatrix:
        """Performs the multiprocess multiplication of matrix sequence

        Returns:
            IMatrix: Result matrix
        """

        # firstly multiplying first and second matrices of the sequence
        # then multiplying each result of previous multiplication with the next matrix in the sequence
        return functools.reduce(
            lambda matrix1, matrix2: matrix_pair_multiplication_command_factory("multiprocess", self._pool, matrix1, matrix2).__call__(), self._matrices)

# this factory creates MultiprocessMatrixSequenceMultiplicationCommand
matrix_sequence_multiplication_command_factory = providers.FactoryAggregate(
    multiprocess=providers.Factory(MultiprocessMatrixSequenceMultiplicationCommand)
)
