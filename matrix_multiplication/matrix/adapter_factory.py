from dependency_injector import providers

from .abc import IMatrix


class MatrixAdapterFactory(providers.FactoryAggregate):
    def map_adapter_to_object(self, matrix_like_class: type, adapter_class: type) -> None:
        self.factories[matrix_like_class.__name__] = providers.Factory(adapter_class)

    def resolve(self, matrix_like_object: object, *args, **kwargs) -> IMatrix:
        return self.factories[matrix_like_object.__class__.__name__](matrix_like_object, *args, **kwargs)
