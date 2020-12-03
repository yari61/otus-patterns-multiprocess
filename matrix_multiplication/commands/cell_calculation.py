import functools
import typing

from dependency_injector import providers


class CalculateMatrixCellValueCommand(typing.Callable):
    """This command calculates the value of the cell based on the first matrix row and the second matrix column
    """
    __slots__ = ("_row", "_column")

    def __init__(self, row: typing.Iterable[typing.SupportsFloat], column: typing.Iterable[typing.SupportsFloat]) -> None:
        self._row = row
        self._column = column

    def __call__(self) -> typing.SupportsFloat:
        """This command calculates the value of the cell based on the first matrix row and the second matrix column

        Raises:
            ValueError: if lengths of the first matrix row and second matrix column are not equal

        Returns:
            typing.SupportsFloat: result matrix cell value
        """

        cell_value = functools.reduce(lambda previous_sum, value_pair: value_pair[0] * value_pair[1] + previous_sum, zip(self._row, self._column), 0)
        return cell_value

# this factory creates CalculateMatrixCellValueCommand
calculate_matrix_cell_value_command_factory = providers.FactoryAggregate(
    default=providers.Factory(CalculateMatrixCellValueCommand)
)
