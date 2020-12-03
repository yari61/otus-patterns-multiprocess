import multiprocessing
import typing

from .commands import CommandContainer
from .matrix.abc import IMatrix


def multiply_matrices(pool: multiprocessing.Pool, matrices: typing.Iterable[IMatrix]) -> IMatrix:
    command_container = CommandContainer()
    validation_command = command_container.validation_command("default", matrices=matrices)
    if not validation_command():
        raise ValueError("matrices could not be multiplied")
    multiplication_command = command_container.multiplication_command("multiprocess", pool=pool, matrices=matrices)
    return multiplication_command()
