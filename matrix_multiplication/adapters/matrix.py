import numpy

from matrix_multiplication.interfaces.matrix import IMatrix


class NDArrayMatrixAdapter(IMatrix):
    """Adapts numpy.ndarray object to IMatrix interface
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
