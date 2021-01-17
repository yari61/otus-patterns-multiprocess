"""Module with abstract classes required for matrix-like objects
"""
from __future__ import annotations
import abc
import typing
from typing import Iterable, List, Tuple


class LeftMultipliableMatrix(abc.ABC):
    """Left matrix in multiplication
    """
    @abc.abstractmethod
    def get_row(self, index: int) -> Iterable[float]:
        pass

    @abc.abstractmethod
    def column_len(self) -> int:
        pass


class RightMultipliableMatrix(abc.ABC):
    """Right matrix in multiplication
    """
    @abc.abstractmethod
    def get_column(self, index: int) -> Iterable[float]:
        pass

    @abc.abstractmethod
    def row_len(self) -> int:
        pass


class ABCMatrix(LeftMultipliableMatrix, RightMultipliableMatrix):
    """Matrix abstract class
    """


class ABCMutableMatrix(ABCMatrix):
    @abc.abstractmethod
    def append(self, row: List[float]) -> None:
        pass

    @abc.abstractmethod
    def reshape(self, new_shape: Tuple[int, int]) -> None:
        pass
