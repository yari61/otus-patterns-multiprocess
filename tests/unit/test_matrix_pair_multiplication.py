import unittest
import random
from unittest.mock import Mock, MagicMock, patch, call, create_autospec

from dependency_injector import providers, containers

from matrix_multiplication.commands.matrix_multiplication import MatrixPairMultiplicationCommand
from tests.utils.generate_matrices import generate_valid_matrix_pair


class CommandContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    command_factory = providers.Factory(
        MatrixPairMultiplicationCommand,
        matrix1=config.matrix1, 
        matrix2=config.matrix2, 
        task_generator_factory=config.task_generator_factory,
        task_processor_factory=config.task_processor_factory,
        matrix_adapter_factory=config.matrix_adapter_factory
    )


class TestMatrixPairMultiplication(unittest.TestCase):
    """This test case checks if the matrix pair multiplication command works correctly
    """
    def test_task_iterator_initialized_once_with_correct_parameters(self):
        # general config
        container = CommandContainer()
        matrix1, matrix2 = generate_valid_matrix_pair()
        task_generator_factory = Mock()
        task_generator_factory.return_value.__iter__ = Mock(return_value=list())
        task_processor_factory = Mock()
        matrix_adapter_factory = Mock()
        container.config.from_dict(
            {
                "matrix1": matrix1,
                "matrix2": matrix2,
                "task_generator_factory": task_generator_factory,
                "task_processor_factory": task_processor_factory,
                "matrix_adapter_factory": matrix_adapter_factory
            }
        )
        command = container.command_factory()
        # command execution
        command()
        # assertion
        task_generator_factory.assert_called_once_with(matrix1=matrix1, matrix2=matrix2)

    def test_task_processor_initialized_once_with_correct_parameters(self):
        # general config
        container = CommandContainer()
        matrix1, matrix2 = generate_valid_matrix_pair()
        task_generator_factory = Mock()
        task_generator_factory.return_value.__iter__ = Mock(return_value=list())
        task_processor_factory = Mock()
        matrix_adapter_factory = Mock()
        container.config.from_dict(
            {
                "matrix1": matrix1,
                "matrix2": matrix2,
                "task_generator_factory": task_generator_factory,
                "task_processor_factory": task_processor_factory,
                "matrix_adapter_factory": matrix_adapter_factory
            }
        )
        command = container.command_factory()
        # command execution
        command()
        # assertion
        task_processor_factory.assert_called_once_with(tasks=task_generator_factory.return_value.__iter__())

    def test_result_matrix_adapter_initialized_once(self):
        # general config
        container = CommandContainer()
        matrix1, matrix2 = generate_valid_matrix_pair()
        task_generator_factory = Mock()
        task_generator_factory.return_value.__iter__ = Mock(return_value=list())
        task_processor_factory = Mock()
        matrix_adapter_factory = Mock()
        container.config.from_dict(
            {
                "matrix1": matrix1,
                "matrix2": matrix2,
                "task_generator_factory": task_generator_factory,
                "task_processor_factory": task_processor_factory,
                "matrix_adapter_factory": matrix_adapter_factory
            }
        )
        command = container.command_factory()
        # command execution
        command()
        # assertion
        matrix_adapter_factory.assert_called_once()

if __name__ == "__main__":
    unittest.main()
