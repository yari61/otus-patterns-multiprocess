import abc
import typing


class ABCMatrix(abc.ABC):
    """Matrix abstract class
    """
    @abc.abstractmethod
    def get_row(self, index: int) -> typing.Iterable[typing.SupportsFloat]:
        pass

    @abc.abstractmethod
    def get_column(self, index: int) -> typing.Iterable[typing.SupportsFloat]:
        pass

    @abc.abstractmethod
    def row_len(self) -> int:
        pass

    @abc.abstractmethod
    def column_len(self) -> int:
        pass
