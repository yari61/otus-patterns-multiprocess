"""Module with IoC containers for commands building
"""
from __future__ import annotations
import numpy
from dependency_injector import containers, providers

from matrix_multiplication.abc import ABCMatrix
from matrix_multiplication.task import MultiprocessTaskProcessor
from matrix_multiplication.matrix.adapters import NDArrayMatrixAdapter
from matrix_multiplication.commands.matrix_multiplication import (
    CalculateCell, BuildTasks, AggregateResult, TaskManager, MultiplyMatrixPair, MultiplyMatrixSequence, ValidateMatrixPair, ValidateMatrixSequence)


class MatrixMultiplicationCommandsContainer(containers.DeclarativeContainer):
    build_tasks = providers.Factory(BuildTasks)
    aggregate_result = providers.Factory(AggregateResult)
    task_manager = providers.Factory(
        TaskManager, build_tasks=build_tasks, aggregate_result=aggregate_result)

    task_processor = providers.Factory(MultiprocessTaskProcessor)

    multiply_matrix_pair = providers.Factory(
        MultiplyMatrixPair, task_manager=task_manager, task_processor=task_processor)
    multiply_matrix_sequence = providers.Factory(
        MultiplyMatrixSequence, multiply_matrix_pair=multiply_matrix_pair)

    validate_matrix_pair = providers.Factory(ValidateMatrixPair)
    validate_matrix_sequence = providers.Factory(
        ValidateMatrixSequence, validate_matrix_pair=validate_matrix_pair)
