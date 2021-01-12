import abc
import typing


class LeftMultipliableMatrix(abc.ABC):
    """Left matrix in multiplication
    """
    @abc.abstractmethod
    def get_row(self, index: int) -> typing.Iterable[typing.SupportsFloat]:
        pass

    @abc.abstractmethod
    def column_len(self) -> int:
        pass


class RightMultipliableMatrix(abc.ABC):
    """Right matrix in multiplication
    """
    @abc.abstractmethod
    def get_column(self, index: int) -> typing.Iterable[typing.SupportsFloat]:
        pass

    @abc.abstractmethod
    def row_len(self) -> int:
        pass


class ABCMatrix(LeftMultipliableMatrix, RightMultipliableMatrix):
    """Matrix abstract class
    """
