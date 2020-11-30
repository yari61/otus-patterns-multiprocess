import functools
import typing


class CalculateCellValueCommand(typing.Callable):
    """This command calculates the value of the cell based on the first matrix row and the second matrix column
    """
    __slots__ = ("_row", "_column")

    def __init__(self, row: typing.Iterable[typing.SupportsFloat], column: typing.Iterable[typing.SupportsFloat]) -> None:
        self._row = row
        self._column = column

    def __call__(self) -> typing.SupportsFloat:
        if len(self._row) != len(self._column):
            raise ValueError(f"row and column lengths are not equal: row {len(self._row)}, column {len(self._column)}")
        cell_value = functools.reduce(lambda previous_sum, value_pair: value_pair[0] * value_pair[1] + previous_sum, zip(self._row, self._column), 0)
        return cell_value
