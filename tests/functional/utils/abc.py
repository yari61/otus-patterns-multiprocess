from __future__ import annotations

import abc
from typing import Tuple, List, Iterable


class ABCMatrixFactory(abc.ABC):
    @abc.abstractmethod
    def __call__(self, shape: Tuple[int, int]) -> ABCMatrix:
        pass


class ABCMatrixSequenceFactory(abc.ABC):
    @abc.abstractmethod
    def __call__(self, shapes: Iterable[Tuple[int, int]]) -> List[ABCMatrix]:
        pass


class ABCShapeSequenceFactory(abc.ABC):
    @abc.abstractmethod
    def __call__(self, length: int) -> List[Tuple[int, int]]:
        pass
