from __future__ import annotations

from dependency_injector import containers, providers

from .abc import ABCMatrixFactory, ABCMatrixSequenceFactory, ABCShapeSequenceFactory
from .matrix import GenerateMatrixSequence


class MatrixContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    matrix_factory = providers.AbstractFactory(ABCMatrixFactory)
    matrix_sequence_factory = providers.AbstractFactory(
        ABCMatrixSequenceFactory)
    shape_sequence_factory = providers.AbstractFactory(ABCShapeSequenceFactory)

    generate_matrix_sequence = providers.Factory(
        GenerateMatrixSequence, matrix_sequence_factory=matrix_sequence_factory, shape_sequence_factory=shape_sequence_factory)
