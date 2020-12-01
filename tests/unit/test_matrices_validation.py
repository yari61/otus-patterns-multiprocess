import unittest
import random

import numpy

from matrix_multiplication.commands import ValidateMatrixSequenceCommand
from matrix_multiplication.adapters import NDArrayMatrixAdapter


def generate_valid_matrix_sequence():
    sequence_len = random.randint(2, 20)
    matrices = [NDArrayMatrixAdapter(numpy.zeros(shape=(random.randint(1, 20), random.randint(1, 20))))]
    for i in range(0, sequence_len):
        matrices.append(NDArrayMatrixAdapter(numpy.zeros(shape=(matrices[-1].row_len(), random.randint(1, 20)))))
    return matrices


def generate_invalid_matrix_sequence():
    sequence_len = random.randint(2, 20)
    matrices = [NDArrayMatrixAdapter(numpy.zeros(shape=(random.randint(1, 20), random.randint(1, 20))))]
    for i in range(0, sequence_len):
        column_len = random.randint(1, 20)
        while column_len == matrices[-1].row_len():
            column_len = random.randint(1, 20)
        matrices.append(NDArrayMatrixAdapter(numpy.zeros(shape=(column_len, random.randint(1, 20)))))
    return matrices


class TestMatricesValidation(unittest.TestCase):
    def test_ndarray_row_len_eq_to_col_len(self):
        matrices = generate_valid_matrix_sequence()
        command = ValidateMatrixSequenceCommand(matrices)
        self.assertTrue(command.__call__())

    def test_ndarray_row_len_ne_to_col_len(self):
        matrices = generate_invalid_matrix_sequence()
        command = ValidateMatrixSequenceCommand(matrices)
        self.assertFalse(command.__call__())

if __name__ == "__main__":
    unittest.main()
