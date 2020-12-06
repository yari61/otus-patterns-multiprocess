import abc
import typing


class Task(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        pass


class TaskGenerator(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> typing.Iterator[Task]:
        pass


class TaskProcessor(abc.ABC):
    @abc.abstractmethod
    def __call__(self) -> typing.Iterable:
        pass
