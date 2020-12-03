import typing

import numpy

from .abc import IMatrix


class OneDimensionalListMatrixAdapter(IMatrix):
    """Adapts one dimensional list object to IMatrix interface
    """
    __slots__ = ("_matrix",)

    def __init__(self, cells: typing.List[typing.SupportsFloat], shape: typing.Tuple[int, int]) -> None:
        matrix = numpy.zeros(shape=shape)
        for row_index in range(0, shape[0]):
            for column_index in range(0, shape[1]):
                matrix[row_index][column_index] = cells[row_index * shape[1] + column_index]
        self._matrix = matrix

    def get_row(self, index: int) -> numpy.ndarray:
        return self._matrix[index, :]

    def get_column(self, index: int) -> numpy.ndarray:
        return self._matrix[:, index]

    def row_len(self) -> int:
        return self._matrix.shape[1]

    def column_len(self) -> int:
        return self._matrix.shape[0]
