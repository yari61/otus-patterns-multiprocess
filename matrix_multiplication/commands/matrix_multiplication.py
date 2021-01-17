"""Module for concrete implementations of classes required to multiply matrix sequence
"""
from __future__ import annotations
import functools
from typing import Iterable, List, Tuple

import numpy as np

from matrix_multiplication.abc.matrix import (
    ABCMatrix, ABCMutableMatrix, LeftMultipliableMatrix, RightMultipliableMatrix)
from matrix_multiplication.abc.task import TaskProcessor
from matrix_multiplication.abc.matrix_multiplication import (
    ABCCalculateCell, ABCBuildTasks, ABCAggregateResult, ABCTaskManager, ABCMultiplyMatrixPair, ABCValidateMatrixPair)
from matrix_multiplication.matrix.adapters import NDArrayMatrixAdapter


class CalculateCell(ABCCalculateCell):
    """Calculates the value of the cell based on the first matrix row and the second matrix column
    """
    __slots__ = ("_row", "_column")

    def __init__(self, row: Iterable[float], column: Iterable[float]) -> None:
        self._row = row
        self._column = column

    def __call__(self) -> float:
        """This command calculates the value of the cell based on the first matrix row and the second matrix column

        Returns:
            float: result matrix cell value
        """

        cell_value = functools.reduce(
            lambda previous_sum, value_pair: value_pair[0] * value_pair[1] + previous_sum, zip(self._row, self._column), 0)
        return cell_value


class BuildTasks(ABCBuildTasks):
    """Builds list of matrix cell calculation tasks
    """
    __slots__ = tuple()

    def __init__(self) -> None:
        pass

    def __call__(self, matrix1: LeftMultipliableMatrix, matrix2: RightMultipliableMatrix) -> List[ABCCalculateCell]:
        tasks = list()
        for row_index in range(0, matrix1.column_len()):
            for column_index in range(0, matrix2.row_len()):
                row, column = matrix1.get_row(row_index), matrix2.get_column(column_index)
                task = self.create_task(row=row, column=column)
                tasks.append(task)
        return tasks

    @staticmethod
    def create_task(row: Iterable[float], column: Iterable[float]) -> ABCCalculateCell:
        return CalculateCell(row=row, column=column)


class AggregateResult(ABCAggregateResult):
    """Aggregates list of task results into matrix of given shape
    """
    __slots__ = tuple()

    def __init__(self):
        pass

    def __call__(self, shape: Tuple[int, int], results: List[float]) -> ABCMutableMatrix:
        results_2d = np.array([results])
        matrix = self.create_matrix(matrix=results_2d)
        matrix.reshape(new_shape=shape)
        return matrix

    @staticmethod
    def create_matrix(matrix: np.ndarray) -> ABCMutableMatrix:
        return NDArrayMatrixAdapter(matrix=matrix)


class TaskManager(ABCTaskManager):
    """Manages tasks builing and results aggregation
    """
    __slots__ = ("_build_tasks", "_aggregate_result")

    def __init__(self, build_tasks: ABCBuildTasks, aggregate_result: ABCAggregateResult) -> None:
        self._build_tasks = build_tasks
        self._aggregate_result = aggregate_result

    def build_tasks(self, matrix1: LeftMultipliableMatrix, matrix2: RightMultipliableMatrix) -> List[ABCCalculateCell]:
        return self._build_tasks(matrix1=matrix1, matrix2=matrix2)

    def handle_results(self, shape, results: Iterable[float]) -> ABCMatrix:
        return self._aggregate_result(shape=shape, results=results)


class MultiplyMatrixPair(ABCMultiplyMatrixPair):
    """Multiplies two matrices
    """
    __slots__ = ("_task_manager", "_task_processor")

    def __init__(self, task_manager: ABCTaskManager, task_processor: TaskProcessor) -> None:
        self._task_manager = task_manager
        self._task_processor = task_processor

    def __call__(self, matrix1: LeftMultipliableMatrix, matrix2: RightMultipliableMatrix) -> ABCMatrix:
        tasks = self._task_manager.build_tasks(
            matrix1=matrix1, matrix2=matrix2)
        results = self._task_processor(tasks=tasks)
        shape = (matrix1.column_len(), matrix2.row_len())
        return self._task_manager.handle_results(shape=shape, results=results)


class MultiplyMatrixSequence(object):
    """Multiplies matrix sequence
    """
    __slots__ = ("_multiply_matrix_pair")

    def __init__(self, multiply_matrix_pair: ABCMultiplyMatrixPair) -> None:
        self._multiply_matrix_pair = multiply_matrix_pair

    def __call__(self, matrices: Iterable[ABCMatrix]) -> ABCMatrix:
        # firstly multiplying first and second matrices of the sequence
        # then multiplying each result of previous multiplication with the next matrix in the sequence
        return functools.reduce(
            lambda matrix1, matrix2: self._multiply_matrix_pair(matrix1=matrix1, matrix2=matrix2), matrices)


class ValidateMatrixPair(ABCValidateMatrixPair):
    """Checks if the matrix pair could be multiplied
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
    """Checks if the sequence of matrices could be multiplied
    """
    __slots__ = ("_validate_matrix_pair",)

    def __init__(self, validate_matrix_pair: ABCValidateMatrixPair) -> None:
        self._validate_matrix_pair = validate_matrix_pair

    def __call__(self, matrices: Iterable[ABCMatrix]) -> bool:
        """Checks if first matrix columns number equals to second matrix rows number for each pair of consecutive matrices in matrix sequence

        Returns:
            bool: True if sequence is valid, else False
        """

        for i in range(0, len(matrices) - 1):
            if not self._validate_matrix_pair(matrix1=matrices[i], matrix2=matrices[i+1]):
                return False
        return True
