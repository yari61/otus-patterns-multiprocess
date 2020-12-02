import typing
import random
from unittest.mock import Mock

from matrix_multiplication.matrix.abc import IMatrix


def generate_mock_matrix(row_len_range=range(0, random.randint(5, 10)), column_len_range=range(0, random.randint(5, 10))) -> IMatrix:
    matrix = Mock()
    matrix.row_len.return_value = random.choice(row_len_range)
    matrix.column_len.return_value = random.choice(column_len_range)
    matrix.get_row.return_value = [0 for i in range(0, matrix.row_len())]
    matrix.get_column.return_value = [0 for i in range(0, matrix.column_len())]
    return matrix


def generate_valid_matrix_pair() -> typing.Tuple[IMatrix, IMatrix]:
    shared_dimension = random.randint(5, 10)
    matrix1 = generate_mock_matrix(row_len_range=[shared_dimension])
    matrix2 = generate_mock_matrix(column_len_range=[shared_dimension])
    return (matrix1, matrix2)


def generate_invalid_matrix_pair() -> typing.Tuple[IMatrix, IMatrix]:
    matrix1 = generate_mock_matrix()
    matrix2 = generate_mock_matrix(column_len_range=[i for i in range(0, random.randint(5, 10)) if i != matrix1.row_len()])
    return (matrix1, matrix2)


def generate_valid_matrix_sequence() -> typing.List[IMatrix]:
    """Generates the sequence of matrices, which could be multiplied

    Returns:
        typing.List[IMatrix]: matrix sequence
    """

    sequence_len = random.randint(2, 20)
    matrices = [generate_mock_matrix()]
    for i in range(0, sequence_len):
        matrices.append(generate_mock_matrix(column_len_range=[matrices[-1].row_len()]))
    return matrices


def generate_invalid_matrix_sequence() -> typing.List[IMatrix]:
    """Generates the sequence of matrices, which could not be multiplied

    Returns:
        typing.List[IMatrix]: matrix sequence
    """

    sequence_len = random.randint(2, 20)
    matrices = [generate_mock_matrix()]
    for i in range(0, sequence_len):
        matrices.append(generate_mock_matrix(column_len_range=[i for i in range(0, random.randint(1, 20)) if i != matrices[-1].row_len()]))
    return matrices
