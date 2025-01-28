import unittest
import pandas as pd
import numpy as np
from TempCalc import (
    get_k_value,
    calculate_min_temp,
    calculate_actual_min_temp,
    calculate_errors
)


class MyTestCase(unittest.TestCase):

    def test_get_k_value(self):
        self.assertEqual(get_k_value(10, 2), -2.2)
        self.assertEqual(get_k_value(15, 4), 0)
        self.assertEqual(get_k_value(30, 6), 0.6)
        self.assertEqual(get_k_value(40, 8), 3.5)
        self.assertIsNone(get_k_value(60, 8))  # Out of bounds test

    def test_calculate_min_temp(self):
        result = calculate_min_temp(20, 10, 10, 2)
        expected = 0.316 * 20 + 0.548 * 10 - 1.24 - 2.2
        self.assertAlmostEqual(result, expected, places=2)

        result_none = calculate_min_temp(20, 10, 60, 2)  # Out of bounds K value
        self.assertIsNone(result_none)

    def test_calculate_actual_min_temp(self):
        result = calculate_actual_min_temp(20, 10, -2.2)
        expected = 0.5 * (20 + 10) - (-2.2)
        self.assertAlmostEqual(result, expected, places=2)

    def test_calculate_errors(self):
        # Prepare a sample dataframe
        data = {
            "Overnight Min Temp": [10, 12, 15],
            "Actual Minimum Temperature": [11, 13, 14]
        }
        df = pd.DataFrame(data)

        # Apply the calculate_errors function
        df = calculate_errors(df)

        # Check the computed errors
        expected_absolute_errors = [1, 1, 1]
        expected_squared_errors = [1, 1, 1]

        self.assertTrue(np.allclose(df["Absolute Error"], expected_absolute_errors))
        self.assertTrue(np.allclose(df["Squared Error"], expected_squared_errors))

if __name__ == '__main__':
    unittest.main()
