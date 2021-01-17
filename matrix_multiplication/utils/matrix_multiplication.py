"""Module with implementations of matrix multiplication functions
"""
from __future__ import annotations
import multiprocessing
import typing

from matrix_multiplication.containers import MatrixMultiplicationCommandsContainer
from matrix_multiplication.abc.matrix import ABCMatrix

commands_container = MatrixMultiplicationCommandsContainer()


def multiprocess_matrices_multiplication(pool: multiprocessing.Pool, matrices: typing.Iterable[ABCMatrix]) -> ABCMatrix:
    commands_container.task_processor.add_kwargs(pool=pool)
    validation_command = commands_container.validate_matrix_sequence()
    if not validation_command(matrices=matrices):
        raise ValueError("matrices could not be multiplied")
    multiplication_command = commands_container.multiply_matrix_sequence()
    return multiplication_command(matrices=matrices)
