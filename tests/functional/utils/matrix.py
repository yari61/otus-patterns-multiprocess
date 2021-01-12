from __future__ import annotations

from typing import final, Tuple, List, Iterable
import random

import numpy as np

from matrix_multiplication.abc import ABCMatrix
from matrix_multiplication.matrix.adapters import NDArrayMatrixAdapter
from .abc import ABCMatrixFactory, ABCMatrixSequenceFactory, ABCShapeSequenceFactory


@final
class RandomMatrixFactory(ABCMatrixFactory):
    def __init__(self) -> None:
        pass

    def __call__(self, shape: Tuple[int, int]) -> ABCMatrix:
        return NDArrayMatrixAdapter(matrix=np.random.rand(*shape))


@final
class ZeroMatrixFactory(ABCMatrixFactory):
    def __init__(self) -> None:
        pass

    def __call__(self, shape: Tuple[int, int]) -> ABCMatrix:
        return NDArrayMatrixAdapter(matrix=np.zeros(shape=shape))


@final
class MatrixSequenceFactory(ABCMatrixSequenceFactory):
    __slots__ = ("_matrix_factory",)

    def __init__(self, matrix_factory: ABCMatrixFactory) -> None:
        self._matrix_factory = matrix_factory

    def __call__(self, shapes: Iterable[Tuple[int, int]]) -> List[ABCMatrix]:
        return [self._matrix_factory(shape) for shape in shapes]


@final
class ValidShapeSequenceFactory(ABCShapeSequenceFactory):
    __slots__ = ("_range",)

    def __init__(self, dim_range: Tuple[int, int] = None) -> None:
        self._range = dim_range if dim_range is not None else (5, 10)

    def __call__(self, length: int) -> List[Tuple[int, int]]:
        shapes = []
        shape = (random.randint(*self._range), random.randint(*self._range))
        shapes.append(shape)
        for i in range(1, length):
            shape = (shape[1], random.randint(*self._range))
            shapes.append(shape)
        return shapes


@final
class InvalidShapeSequenceFactory(ABCShapeSequenceFactory):
    __slots__ = ("_error_count_range", "_valid_shape_sequence_factory")

    def __init__(self, error_count_range: Tuple[int, int] = None, valid_shape_sequence_factory: ABCShapeSequenceFactory = None) -> None:
        self._error_count_range = error_count_range if error_count_range is not None else (1, 3)
        self._valid_shape_sequence_factory = (
            valid_shape_sequence_factory if valid_shape_sequence_factory is not None else ValidShapeSequenceFactory())

    def __call__(self, length: int) -> List[Tuple[int, int]]:
        errors_count = random.randint(*self._error_count_range)
        error_location = 0
        shapes = list()
        for i in range(errors_count):
            step = random.randint(
                1, length - error_location + i - errors_count)
            shapes.extend(self._valid_shape_sequence_factory(length=step))
            error_location = error_location + step
        return shapes


@final
class GenerateMatrixSequence(object):
    __slots__ = ("_matrix_sequence_factory", "_shape_sequence_factory")

    def __init__(self, matrix_sequence_factory: ABCMatrixSequenceFactory, shape_sequence_factory: ABCShapeSequenceFactory) -> None:
        self._matrix_sequence_factory = matrix_sequence_factory
        self._shape_sequence_factory = shape_sequence_factory

    def __call__(self, length: int) -> List[ABCMatrix]:
        shapes = self._shape_sequence_factory(length=length)
        return self._matrix_sequence_factory(shapes=shapes)
