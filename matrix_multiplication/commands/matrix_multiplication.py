import functools
import typing

from dependency_injector import providers

from matrix_multiplication.abc.matrix import ABCMatrix
from matrix_multiplication.abc.task import Task, TaskGenerator, TaskProcessor


class CalculateMatrixCellValueCommand(Task):
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


class MatrixPairMultiplicationTaskGenerator(TaskGenerator):
    """Builds tasks of calculation of each cell of a result matrix for matrix pair multiplication
    """
    __slots__ = ("_matrix1", "_matrix2", "_task_factory")

    def __init__(
        self, 
        matrix1: ABCMatrix, 
        matrix2: ABCMatrix, 
        task_factory: providers.Factory
    ) -> None:

        self._matrix1 = matrix1
        self._matrix2 = matrix2
        self._task_factory = task_factory

    def __iter__(self) -> typing.Iterator[Task]:
        """Iterates over cell value calculation tasks in order from the upper left cell to the lower right walking through rows

        Returns:
            Task: cell value calculation task

        Yields:
            Iterator[Task]: cell value calculation tasks
        """

        for row_index in range(0, self._matrix1.column_len()):
            for column_index in range(0, self._matrix2.row_len()):
                yield self._task_factory(row=self._matrix1.get_row(row_index), column=self._matrix2.get_column(column_index))


class MatrixPairMultiplicationCommand(Task):
    """Performs the multiprocess multiplication of two matrices
    """
    __slots__ = ("_matrix1", "_matrix2", "_task_generator_factory", "_task_processor_factory", "_result_matrix_adapter_factory")

    def __init__(
        self,
        matrix1: ABCMatrix,
        matrix2: ABCMatrix,
        task_generator_factory: providers.Factory,
        task_processor_factory: providers.Factory,
        matrix_adapter_factory: providers.Factory
    ) -> None:

        self._matrix1 = matrix1
        self._matrix2 = matrix2
        self._task_generator_factory = task_generator_factory
        self._task_processor_factory = task_processor_factory
        self._result_matrix_adapter_factory = matrix_adapter_factory

    def __call__(self) -> ABCMatrix:
        """Performs the multiprocess multiplication of two matrices

        Returns:
            ABCMatrix: Result matrix
        """

        task_generator: TaskGenerator = self._task_generator_factory(matrix1=self._matrix1, matrix2=self._matrix2)
        task_processor: TaskProcessor = self._task_processor_factory(tasks=[task for task in task_generator.__iter__()])
        task_results = task_processor.__call__()
        shape: typing.Tuple[int, int] = (self._matrix1.column_len(), self._matrix2.row_len())
        return self._result_matrix_adapter_factory(cells=task_results, shape=shape)


class MatrixSequenceMultiplicationCommand(Task):
    """Performs the multiprocess multiplication of matrix sequence
    """
    __slots__ = ("_matrices", "_matrix_pair_multiplication_command_factory")

    def __init__(
        self,
        matrices: typing.Iterable[ABCMatrix],
        matrix_pair_multiplication_command_factory: providers.Factory
    ) -> None:

        self._matrices = matrices
        self._matrix_pair_multiplication_command_factory = matrix_pair_multiplication_command_factory

    def __call__(self) -> ABCMatrix:
        """Performs the multiprocess multiplication of matrix sequence

        Returns:
            ABCMatrix: Result matrix
        """

        # firstly multiplying first and second matrices of the sequence
        # then multiplying each result of previous multiplication with the next matrix in the sequence
        return functools.reduce(
            lambda matrix1, matrix2: self._matrix_pair_multiplication_command_factory.__call__(matrix1=matrix1, matrix2=matrix2).__call__(), self._matrices)


class ValidateMatrixPairCommand(Task):
    """This command checks if the matrix pair could be multiplied
    """
    __slots__ = ("_matrix1", "_matrix2")

    def __init__(self, matrix1: ABCMatrix, matrix2: ABCMatrix) -> None:
        self._matrix1 = matrix1
        self._matrix2 = matrix2

    def __call__(self) -> bool:
        """Checks if first matrix columns number equals to second matrix rows number for each pair of consecutive matrices in matrix sequence

        Returns:
            bool: True if pair could be multiplied, else False
        """

        if self._matrix1.row_len() != self._matrix2.column_len():
            return False
        return True


class ValidateMatrixSequenceCommand(Task):
    """This command checks if the sequence of matrices could be multiplied
    """
    __slots__ = ("_matrices", "_validate_matrix_pair_command_factory")

    def __init__(
        self, 
        matrices: typing.Iterable[ABCMatrix], 
        validate_matrix_pair_command_factory: providers.Factory
    ) -> None:

        self._matrices = matrices
        self._validate_matrix_pair_command_factory = validate_matrix_pair_command_factory

    def __call__(self) -> bool:
        """Checks if first matrix columns number equals to second matrix rows number for each pair of consecutive matrices in matrix sequence

        Returns:
            bool: True if sequence is valid, else False
        """

        for i in range(0, len(self._matrices) - 1):
            validation_command: Task = self._validate_matrix_pair_command_factory(matrix1=self._matrices[i], matrix2=self._matrices[i+1])
            if not validation_command.__call__():
                return False
        return True
