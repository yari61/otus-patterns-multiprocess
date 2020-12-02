from dependency_injector import containers

import numpy

from .adapter_ndarray import NDArrayMatrixAdapter
from .adapter_factory import MatrixAdapterFactory


class MatrixAdapterContainer(containers.DeclarativeContainer):
    matrix_adapter_factory = MatrixAdapterFactory()

    matrix_adapter_factory.map_adapter_to_object(numpy.ndarray, NDArrayMatrixAdapter)
