from dependency_injector import containers

from .validation import validate_matrix_sequence_command_factory
from .matrix_multiplication import matrix_sequence_multiplication_command_factory


class CommandContainer(containers.DeclarativeContainer):
    validation_command = validate_matrix_sequence_command_factory
    multiplication_command = matrix_sequence_multiplication_command_factory

__all__ = ["CommandContainer"]
