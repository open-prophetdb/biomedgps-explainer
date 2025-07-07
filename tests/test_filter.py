import unittest
import os
import tempfile
import pandas as pd
from unittest.mock import patch, mock_open
from drugs4disease.filter import DrugFilter


class TestDrugFilter(unittest.TestCase):
    """Test cases for DrugFilter class"""

    def setUp(self):
        """Set up test fixtures"""
        self.filter = DrugFilter()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data = {
            'drug_id': ['MESH:C001', 'MESH:C002', 'MESH:C003', 'MESH:C004'],
            'drug_name': ['Drug1', 'Drug2', 'Drug3', 'Drug4'],
            'score': [0.9, 0.7, 0.5, 0.3],
            'pvalue': [0.01, 0.05, 0.1, 0.2],
            'num_of_shared_genes': [5, 3, 2, 1],
            'existing': [True, False, False, True]
        }
        self.test_df = pd.DataFrame(self.test_data)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test DrugFilter initialization"""
        self.assertIsNotNone(self.filter)

    def test_parse_expression_simple(self):
        """Test parsing simple expression"""
        expression = "score > 0.5"
        result = DrugFilter.parse_expression(expression)
        self.assertEqual(result, "score > 0.5")

    def test_parse_expression_with_and(self):
        """Test parsing expression with && operator"""
        expression = "score > 0.5 && num_of_shared_genes >= 2"
        result = DrugFilter.parse_expression(expression)
        self.assertEqual(result, "score > 0.5 and num_of_shared_genes >= 2")

    def test_parse_expression_with_or(self):
        """Test parsing expression with || operator"""
        expression = "score > 0.8 || existing == True"
        result = DrugFilter.parse_expression(expression)
        self.assertEqual(result, "score > 0.8 or existing == True")

    def test_parse_expression_complex(self):
        """Test parsing complex expression with multiple operators"""
        expression = "score > 0.5 && pvalue < 0.1 || existing == True"
        result = DrugFilter.parse_expression(expression)
        self.assertEqual(result, "score > 0.5 and pvalue < 0.1 or existing == True")

    def test_filter_dataframe_simple_condition(self):
        """Test filtering DataFrame with simple condition"""
        result = DrugFilter.filter_dataframe(self.test_df, "score > 0.5")
        self.assertEqual(len(result), 2)  # Only 2 drugs have score > 0.5
        self.assertTrue(all(result['score'] > 0.5))

    def test_filter_dataframe_complex_condition(self):
        """Test filtering DataFrame with complex condition"""
        result = DrugFilter.filter_dataframe(self.test_df, "score > 0.5 and num_of_shared_genes >= 2")
        self.assertEqual(len(result), 2)  # Drugs with score > 0.5 AND shared_genes >= 2
        self.assertTrue(all(result['score'] > 0.5))
        self.assertTrue(all(result['num_of_shared_genes'] >= 2))

    def test_filter_dataframe_or_condition(self):
        """Test filtering DataFrame with OR condition"""
        result = DrugFilter.filter_dataframe(self.test_df, "score > 0.8 or existing == True")
        self.assertEqual(len(result), 2)  # Drugs with score > 0.8 OR existing == True
        self.assertTrue(all((result['score'] > 0.8) | (result['existing'] == True)))

    def test_filter_dataframe_not_condition(self):
        """Test filtering DataFrame with NOT condition"""
        result = DrugFilter.filter_dataframe(self.test_df, "not existing == True")
        self.assertEqual(len(result), 2)  # Drugs where existing is not True
        self.assertTrue(all(result['existing'] == False))

    def test_filter_dataframe_invalid_column(self):
        """Test filtering DataFrame with invalid column"""
        with self.assertRaises(Exception):  # pandas raises UndefinedVariableError
            DrugFilter.filter_dataframe(self.test_df, "invalid_column > 0.5")

    def test_filter_dataframe_invalid_operator(self):
        """Test filtering DataFrame with invalid operator"""
        with self.assertRaises(Exception):  # pandas.query will raise an exception
            DrugFilter.filter_dataframe(self.test_df, "score INVALID 0.5")

    def test_filter_dataframe_no_matches(self):
        """Test filtering DataFrame with no matches"""
        result = DrugFilter.filter_dataframe(self.test_df, "score > 1.0")
        self.assertEqual(len(result), 0)

    def test_filter_dataframe_all_matches(self):
        """Test filtering DataFrame with all matches"""
        result = DrugFilter.filter_dataframe(self.test_df, "score >= 0.3")
        self.assertEqual(len(result), 4)

    def test_filter_drugs_file_operation(self):
        """Test filtering drugs with file input/output"""
        # Create input Excel file
        input_file = os.path.join(self.temp_dir, 'input.xlsx')
        output_file = os.path.join(self.temp_dir, 'output.xlsx')
        
        with pd.ExcelWriter(input_file) as writer:
            self.test_df.to_excel(writer, sheet_name='annotated_drugs', index=False)
        
        # Test filtering
        self.filter.filter_drugs(input_file, "score > 0.5", output_file)
        
        # Verify output file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Verify content - should have both sheets
        with pd.ExcelFile(output_file) as xls:
            self.assertIn('annotated_drugs', xls.sheet_names)
            self.assertIn('filtered_drugs', xls.sheet_names)
            
            # Check original data
            original_df = pd.read_excel(output_file, sheet_name='annotated_drugs')
            pd.testing.assert_frame_equal(original_df, self.test_df)
            
            # Check filtered data
            filtered_df = pd.read_excel(output_file, sheet_name='filtered_drugs')
            self.assertEqual(len(filtered_df), 2)  # Only 2 drugs have score > 0.5
            self.assertTrue(all(filtered_df['score'] > 0.5))

    def test_filter_drugs_custom_sheet_names(self):
        """Test filtering drugs with custom sheet names"""
        # Create input Excel file
        input_file = os.path.join(self.temp_dir, 'input.xlsx')
        output_file = os.path.join(self.temp_dir, 'output.xlsx')
        
        with pd.ExcelWriter(input_file) as writer:
            self.test_df.to_excel(writer, sheet_name='drugs', index=False)
        
        # Test filtering with custom sheet names
        self.filter.filter_drugs(input_file, "score > 0.5", output_file, ("drugs", "filtered"))
        
        # Verify output file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Verify content - should have both sheets with custom names
        with pd.ExcelFile(output_file) as xls:
            self.assertIn('drugs', xls.sheet_names)
            self.assertIn('filtered', xls.sheet_names)
            
            # Check filtered data
            filtered_df = pd.read_excel(output_file, sheet_name='filtered')
            self.assertEqual(len(filtered_df), 2)
            self.assertTrue(all(filtered_df['score'] > 0.5))

    def test_filter_drugs_no_matches_file(self):
        """Test filtering drugs with no matches in file operation"""
        # Create input Excel file
        input_file = os.path.join(self.temp_dir, 'input.xlsx')
        output_file = os.path.join(self.temp_dir, 'output.xlsx')
        
        with pd.ExcelWriter(input_file) as writer:
            self.test_df.to_excel(writer, sheet_name='annotated_drugs', index=False)
        
        # Test filtering with condition that matches nothing
        self.filter.filter_drugs(input_file, "score > 1.0", output_file)
        
        # Verify output file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check filtered data should be empty
        filtered_df = pd.read_excel(output_file, sheet_name='filtered_drugs')
        self.assertEqual(len(filtered_df), 0)

    def test_filter_drugs_file_not_found(self):
        """Test filtering drugs with non-existent input file"""
        input_file = os.path.join(self.temp_dir, 'nonexistent.xlsx')
        output_file = os.path.join(self.temp_dir, 'output.xlsx')
        
        with self.assertRaises(FileNotFoundError):
            self.filter.filter_drugs(input_file, "score > 0.5", output_file)

    def test_filter_drugs_invalid_expression(self):
        """Test filtering drugs with invalid expression"""
        # Create input Excel file
        input_file = os.path.join(self.temp_dir, 'input.xlsx')
        output_file = os.path.join(self.temp_dir, 'output.xlsx')
        
        with pd.ExcelWriter(input_file) as writer:
            self.test_df.to_excel(writer, sheet_name='annotated_drugs', index=False)
        
        # Test filtering with invalid expression
        with self.assertRaises(Exception):  # pandas.query will raise an exception
            self.filter.filter_drugs(input_file, "invalid expression", output_file)

    def test_static_methods_independence(self):
        """Test that static methods work independently of instance"""
        # Test parse_expression as static method
        result1 = DrugFilter.parse_expression("score > 0.5 && pvalue < 0.1")
        result2 = self.filter.parse_expression("score > 0.5 && pvalue < 0.1")
        self.assertEqual(result1, result2)
        
        # Test filter_dataframe as static method
        result1 = DrugFilter.filter_dataframe(self.test_df, "score > 0.5")
        result2 = self.filter.filter_dataframe(self.test_df, "score > 0.5")
        pd.testing.assert_frame_equal(result1, result2)


if __name__ == '__main__':
    unittest.main() 