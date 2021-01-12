"""This module contains all commands required to multiply matrix sequence
"""
import functools
import typing

from dependency_injector import providers

from matrix_multiplication.abc.matrix import ABCMatrix
from matrix_multiplication.abc.task import Task, TaskGenerator, TaskProcessor


class CalculateMatrixCellValueCommand(Task):
    """This command calculates the value of the cell based on the first matrix row and the second matrix column
    """
    __slots__ = ("_row", "_column")

    def __init__(self, row: typing.Iterable[typing.SupportsFloat], column: typing.Iterable[typing.SupportsFloat]) -> None:
        self._row = row
        self._column = column

    def __call__(self) -> typing.SupportsFloat:
        """This command calculates the value of the cell based on the first matrix row and the second matrix column

        Returns:
            typing.SupportsFloat: result matrix cell value
        """

        cell_value = functools.reduce(
            lambda previous_sum, value_pair: value_pair[0] * value_pair[1] + previous_sum, zip(self._row, self._column), 0)
        return cell_value


class MatrixPairMultiplicationTaskGenerator(TaskGenerator):
    """Generates calculation tasks of each cell of a result matrix for matrix pair multiplication
    """
    __slots__ = ("_matrix1", "_matrix2", "_task_factory")

    def __init__(
        self,
        matrix1: ABCMatrix,
        matrix2: ABCMatrix,
        task_factory: providers.Factory
    ) -> None:

        self._matrix1 = matrix1
        self._matrix2 = matrix2
        self._task_factory = task_factory

    def __iter__(self) -> typing.Iterator[Task]:
        """Iterates over cell value calculation tasks in order from the upper left cell to the lower right walking through rows

        Returns:
            Task: cell value calculation task

        Yields:
            Iterator[Task]: cell value calculation tasks
        """

        for row_index in range(0, self._matrix1.column_len()):
            for column_index in range(0, self._matrix2.row_len()):
                yield self._task_factory(row=self._matrix1.get_row(row_index), column=self._matrix2.get_column(column_index))


class ABCMultiplyMatrixPair:
    def __call__(self, matrix1: ABCMatrix, matrix2: ABCMatrix) -> ABCMatrix:
        pass


class MultiplyMatrixPair(ABCMultiplyMatrixPair):
    """Performs the multiplication of two matrices
    """
    __slots__ = ("_task_generator_factory",
                 "_task_processor_factory", "_result_matrix_adapter_factory")

    def __init__(self, task_generator_factory: providers.Factory, task_processor_factory: providers.Factory, matrix_adapter_factory: providers.Factory) -> None:
        self._task_generator_factory = task_generator_factory
        self._task_processor_factory = task_processor_factory
        self._result_matrix_adapter_factory = matrix_adapter_factory

    def __call__(self, matrix1: ABCMatrix, matrix2: ABCMatrix) -> ABCMatrix:
        """Performs the multiprocess multiplication of two matrices

        Returns:
            ABCMatrix: Result matrix
        """

        task_generator: TaskGenerator = self._task_generator_factory(
            matrix1=matrix1, matrix2=matrix2)
        task_processor: TaskProcessor = self._task_processor_factory(
            tasks=[task for task in task_generator.__iter__()])
        task_results = task_processor.__call__()
        shape: typing.Tuple[int, int] = (
            matrix1.column_len(), matrix2.row_len())
        return self._result_matrix_adapter_factory(cells=task_results, shape=shape)


class MultiplyMatrixSequence(object):
    """Performs the multiplication of matrix sequence
    """
    __slots__ = ("_multiply_matrix_pair")

    def __init__(self, multiply_matrix_pair: ABCMultiplyMatrixPair) -> None:
        self._multiply_matrix_pair = multiply_matrix_pair

    def __call__(self, matrices: typing.Iterable[ABCMatrix]) -> ABCMatrix:
        """Performs the multiprocess multiplication of matrix sequence

        Returns:
            ABCMatrix: Result matrix
        """

        # firstly multiplying first and second matrices of the sequence
        # then multiplying each result of previous multiplication with the next matrix in the sequence
        return functools.reduce(
            lambda matrix1, matrix2: self._multiply_matrix_pair(matrix1=matrix1, matrix2=matrix2), matrices)


class ABCValidateMatrixPair:
    def __call__(self, matrix1: ABCMatrix, matrix2: ABCMatrix) -> bool:
        pass


class ValidateMatrixPair(ABCValidateMatrixPair):
    """This command checks if the matrix pair could be multiplied
    """
    __slots__ = tuple()

    def __init__(self) -> None:
        pass

    def __call__(self, matrix1: ABCMatrix, matrix2: ABCMatrix) -> bool:
        """Checks if first matrix columns number equals to second matrix rows number for each pair of consecutive matrices in matrix sequence

        Returns:
            bool: True if pair could be multiplied, else False
        """

        if matrix1.row_len() != matrix2.column_len():
            return False
        return True


class ValidateMatrixSequence(object):
    """This command checks if the sequence of matrices could be multiplied
    """
    __slots__ = ("_validate_matrix_pair",)

    def __init__(self, validate_matrix_pair: ABCValidateMatrixPair) -> None:
        self._validate_matrix_pair = validate_matrix_pair

    def __call__(self, matrices: typing.Iterable[ABCMatrix]) -> bool:
        """Checks if first matrix columns number equals to second matrix rows number for each pair of consecutive matrices in matrix sequence

        Returns:
            bool: True if sequence is valid, else False
        """

        for i in range(0, len(matrices) - 1):
            if not self._validate_matrix_pair(matrix1=matrices[i], matrix2=matrices[i+1]):
                return False
        return True
