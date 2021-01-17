"""Module with abstract classes required for objects related with tasks
"""
from __future__ import annotations
import abc
import typing


class Task(abc.ABC):
    @abc.abstractmethod
    def __call__(self):
        pass


class TaskProcessor(abc.ABC):
    @abc.abstractmethod
    def __call__(self) -> typing.Iterable:
        pass
