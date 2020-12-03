import collections

from dependency_injector import containers

import numpy

from .adapter_factory import MatrixAdapterFactory
from .adapter_ndarray import NDArrayMatrixAdapter
from .adapter_one_dimensional_list import OneDimensionalListMatrixAdapter


class MatrixAdapterContainer(containers.DeclarativeContainer):
    matrix_adapter_factory = MatrixAdapterFactory()

    matrix_adapter_factory.map_adapter_to_object(numpy.ndarray, NDArrayMatrixAdapter)
    matrix_adapter_factory.map_adapter_to_object(collections.deque, OneDimensionalListMatrixAdapter)
