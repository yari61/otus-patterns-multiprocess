import typing

import numpy

from matrix_multiplication.interfaces.matrix import IMatrix


class ValidateMatricesCommand(typing.Callable):
    """This command checks if the sequence of matrices could be multiplied
    """
    __slots__ = ("_matrices",)

    def __init__(self, *matrices: typing.List[IMatrix]) -> None:
        self._matrices = matrices

    def __call__(self) -> bool:
        """Checks if first matrix columns count equals to second matrix rows count for each pair of consecutive matrices

        Returns:
            bool: True if sequence is valid, else False
        """

        for i in range(0, len(self._matrices) - 1):
            if self._matrices[i].row_len() != self._matrices[i+1].column_len():
                return False
        return True
