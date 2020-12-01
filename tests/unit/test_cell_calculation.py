import unittest
import random
import math

from matrix_multiplication.commands import CalculateMatrixCellValueCommand


class TestCellCalculation(unittest.TestCase):
    """This test case checks if the calculation of any matrix cell works correctly
    """
    def test_calculation(self):
        # generating random first matrix row and second matrix column length
        n = random.randint(10, 20)
        # generating random row
        row = [random.random() for i in range(0, n)]
        # generating random column
        column = [random.random() for i in range(0, n)]
        # creating command for cell value calculation
        command = CalculateMatrixCellValueCommand(row=row, column=column)
        # calculating expected value
        expected_value = sum([row[i] * column[i] for i in range(0, n)])
        self.assertAlmostEqual(expected_value, command.__call__())

if __name__ == "__main__":
    unittest.main()
