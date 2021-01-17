"""Module with abstract classes required for matrices multiplication
"""
from __future__ import annotations
import abc
from typing import Iterable, List, Tuple

from .matrix import (
    ABCMatrix, LeftMultipliableMatrix, RightMultipliableMatrix)
from .task import Task


class ABCCalculateCell(Task):
    @abc.abstractmethod
    def __init__(self, row: Iterable[float], column: Iterable[float]) -> None:
        pass


class ABCBuildTasks(abc.ABC):
    @abc.abstractmethod
    def __call__(self, matrix1: LeftMultipliableMatrix, matrix2: RightMultipliableMatrix) -> List[Task]:
        pass


class ABCAggregateResult(abc.ABC):
    @abc.abstractmethod
    def __call__(self, shape: Tuple[int, int], results: List[float]) -> ABCMatrix:
        pass


class ABCTaskManager(abc.ABC):
    @abc.abstractmethod
    def build_tasks(self, matrix1: LeftMultipliableMatrix, matrix2: RightMultipliableMatrix) -> List[ABCCalculateCell]:
        pass

    @abc.abstractmethod
    def handle_results(self, results: Iterable[float]) -> ABCMatrix:
        pass


class ABCMultiplyMatrixPair:
    @abc.abstractmethod
    def __call__(self, matrix1: LeftMultipliableMatrix, matrix2: RightMultipliableMatrix) -> ABCMatrix:
        pass


class ABCValidateMatrixPair:
    @abc.abstractmethod
    def __call__(self, matrix1: ABCMatrix, matrix2: ABCMatrix) -> bool:
        pass
