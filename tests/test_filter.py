import unittest
import pandas as pd
from drugs4disease.filter import DrugFilter

class TestDrugFilter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'score': [0.6, 0.4, 0.8],
            'status': ['known', 'unknown', 'known']
        })

    def test_parse_expression(self):
        expr = 'score >= 0.5 && status != "unknown"'
        parsed = DrugFilter.parse_expression(expr)
        self.assertIn('and', parsed)
        self.assertIn('!=', parsed)

    def test_filter_dataframe(self):
        expr = 'score >= 0.5 && status != "unknown"'
        filtered = DrugFilter.filter_dataframe(self.df, expr)
        self.assertEqual(filtered.shape[0], 2)

if __name__ == '__main__':
    unittest.main() 