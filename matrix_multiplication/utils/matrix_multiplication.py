import multiprocessing
import typing

from matrix_multiplication.containers import MatrixMultiplicationCommandsContainer
from matrix_multiplication.abc.matrix import ABCMatrix

commands_container = MatrixMultiplicationCommandsContainer()


def multiprocess_matrices_multiplication(pool: multiprocessing.Pool, matrices: typing.Iterable[ABCMatrix]) -> ABCMatrix:
    commands_container.task_processor_factory.add_kwargs(pool=pool)
    validation_command = commands_container.validate_sequence_factory()
    if not validation_command(matrices=matrices):
        raise ValueError("matrices could not be multiplied")
    multiplication_command = commands_container.sequence_multiplication_factory(matrices=matrices)
    return multiplication_command()

if __name__ == "__main__":
    pass
