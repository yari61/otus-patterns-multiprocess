"""Module with matrix adapters
"""
from __future__ import annotations
from typing import List, Tuple

import numpy

from matrix_multiplication.abc.matrix import ABCMatrix, ABCMutableMatrix


def to_ndarray(matrix: ABCMatrix) -> numpy.ndarray:
    """Converts :class:`ABCMatrix` to :class:`numpy.ndarray`

    Args:
        matrix (ABCMatrix): Matrix object to convert

    Returns:
        numpy.ndarray: Converted object
    """
    rows = list()
    for i in range(0, matrix.column_len()):
        row = matrix.get_row(i)
        rows.append(row)
    return numpy.array(rows)


class NDArrayMatrixAdapter(ABCMutableMatrix):
    """Adapts numpy.ndarray object to ABCMatrix interface
    """
    __slots__ = ("_matrix",)

    def __init__(self, matrix: numpy.ndarray) -> None:
        if matrix.ndim != 2:
            raise ValueError("the given object is not a matrix")
        self._matrix = matrix

    def get_row(self, index: int) -> numpy.ndarray:
        return self._matrix[index, :]

    def get_column(self, index: int) -> numpy.ndarray:
        return self._matrix[:, index]

    def row_len(self) -> int:
        return self._matrix.shape[1]

    def column_len(self) -> int:
        return self._matrix.shape[0]

    def append(self, row: List[float]) -> None:
        self._matrix = numpy.append(self._matrix, row)

    def reshape(self, new_shape: Tuple[int, int]) -> None:
        self._matrix = self._matrix.reshape(new_shape)
