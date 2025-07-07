import unittest
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import patch, Mock
from drugs4disease.visualizer import Visualizer
import numpy as np


class TestVisualizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.disease_id = "MESH:D001"
        self.disease_name = "Test Disease"
        self.visualizer = Visualizer(
            disease_id=self.disease_id,
            disease_name=self.disease_name
        )
        
        # Create test data
        self.test_data = {
            'drug_id': ['MESH:C001', 'MESH:C002', 'MESH:C003', 'MESH:C004'],
            'drug_name': ['Drug1', 'Drug2', 'Drug3', 'Drug4'],
            'score': [0.9, 0.7, 0.5, 0.3],
            'pvalue': [0.01, 0.05, 0.1, 0.2],
            'num_of_shared_genes_in_path': [5, 3, 2, 1],
            'num_of_shared_pathways': [3, 2, 1, 0],
            'num_of_shared_genes': [4, 2, 1, 0],
            'existing': [True, False, False, True],
            'drug_degree': [15, 8, 5, 2],
            'num_of_key_genes': [3, 2, 1, 0],
            'key_gene_names': ['Gene1,Gene2,Gene3', 'Gene1,Gene2', 'Gene1', ''],
            'shared_gene_names': ['Gene1,Gene2,Gene3,Gene4', 'Gene1,Gene2', 'Gene1', ''],
            'shared_pathways': ['Pathway1,Pathway2,Pathway3', 'Pathway1,Pathway2', 'Pathway1', ''],
            'shared_disease_names': ['Disease1,Disease2', 'Disease1', '', '']
        }
        self.test_df = pd.DataFrame(self.test_data)
        
        # Create filtered data (subset of test data)
        self.filtered_data = {
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2'],
            'score': [0.9, 0.7],
            'pvalue': [0.01, 0.05],
            'num_of_shared_genes_in_path': [5, 3],
            'num_of_shared_pathways': [3, 2],
            'num_of_shared_genes': [4, 2],
            'existing': [True, False],
            'drug_degree': [15, 8],
            'num_of_key_genes': [3, 2],
            'key_gene_names': ['Gene1,Gene2,Gene3', 'Gene1,Gene2'],
            'shared_gene_names': ['Gene1,Gene2,Gene3,Gene4', 'Gene1,Gene2'],
            'shared_pathways': ['Pathway1,Pathway2,Pathway3', 'Pathway1,Pathway2'],
            'shared_disease_names': ['Disease1,Disease2', 'Disease1']
        }
        self.filtered_df = pd.DataFrame(self.filtered_data)
        
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test Excel file with required sheets
        self.test_file = os.path.join(self.temp_dir, 'test_data.xlsx')
        df = pd.DataFrame(self.test_data)
        filtered_df = df[df['score'] > 0.5]  # Filtered data
        
        with pd.ExcelWriter(self.test_file) as writer:
            df.to_excel(writer, sheet_name='annotated_drugs', index=False)
            filtered_df.to_excel(writer, sheet_name='filtered_drugs', index=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test Visualizer initialization"""
        self.assertIsNotNone(self.visualizer)
    
    def test_init_with_embed_images_false(self):
        """Test Visualizer initialization with embed_images=False"""
        viz = Visualizer(
            disease_id="MESH:D001",
            disease_name="Test Disease",
            embed_images=False
        )
        self.assertFalse(viz.embed_images)
    
    def test_get_chart_types(self):
        """Test getting chart types"""
        chart_types = Visualizer.get_chart_types()
        expected_types = [
            "score_distribution",
            "predicted_score_boxplot",
            "disease_similarity_heatmap",
            "network_centrality",
            "shared_genes_pathways",
            "drug_disease_network",
            "shared_gene_count",
            "score_vs_degree",
            "shared_gene_count_vs_score",
            "overlap_pathways",
            "key_genes_distribution",
            "existing_vs_predicted",
            "prompt",
        ]
        self.assertEqual(set(chart_types), set(expected_types))
    
    def test_get_chart_title(self):
        """Test getting chart titles"""
        title = Visualizer.get_chart_title("score_distribution")
        self.assertEqual(title, "Drug Predicted Score Distribution")
        
        title = Visualizer.get_chart_title("shared_gene_count")
        self.assertEqual(title, "Number of Shared Genes Between Drugs and Diseases")
        
        # Test unknown chart type
        title = Visualizer.get_chart_title("unknown_chart")
        self.assertEqual(title, "unknown_chart")
    
    def test_plot_score_distribution(self):
        """Test score distribution plotting"""
        output_file = os.path.join(self.temp_dir, 'score_distribution.png')
        
        interpretation = self.visualizer.plot_score_distribution(
            self.test_df, self.filtered_df, output_file
        )
        
        # Check if file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check if interpretation contains expected text
        self.assertIn('score distribution', interpretation.lower())
        self.assertIn('histogram', interpretation.lower())
    
    def test_plot_predicted_score_boxplot(self):
        """Test predicted score boxplot plotting"""
        output_file = os.path.join(self.temp_dir, 'predicted_score_boxplot.png')
        
        interpretation = self.visualizer.plot_predicted_score_boxplot(
            self.test_df, self.filtered_df, output_file
        )
        
        # Check if file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check if interpretation contains expected text
        self.assertIn('predicted score distribution', interpretation.lower())
        self.assertIn('box plot', interpretation.lower())
    
    def test_plot_shared_genes_pathways(self):
        """Test shared genes pathways plotting"""
        output_file = os.path.join(self.temp_dir, 'shared_genes_pathways.png')
        
        interpretation = self.visualizer.plot_shared_genes_pathways(
            self.test_df, self.filtered_df, output_file
        )
        
        # Check if file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check if interpretation contains expected text
        self.assertIn('shared genes', interpretation.lower())
        self.assertIn('pathways', interpretation.lower())
    
    def test_plot_network_centrality(self):
        """Test network centrality plotting"""
        output_file = os.path.join(self.temp_dir, 'network_centrality.png')
        
        interpretation = self.visualizer.plot_network_centrality(
            self.test_df, self.filtered_df, output_file
        )
        
        # Check if file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check if interpretation contains expected text
        self.assertIn('network centrality', interpretation.lower())
    
    def test_plot_score_distribution_empty_data(self):
        """Test score distribution plotting with empty data"""
        empty_df = pd.DataFrame()
        output_file = os.path.join(self.temp_dir, 'empty_score_dist.png')
        
        # Should handle empty DataFrame gracefully
        with self.assertRaises(ValueError):
            self.visualizer.plot_score_distribution(
                empty_df, empty_df, output_file
            )
    
    def test_create_visualization_score_distribution(self):
        """Test creating score distribution visualization"""
        output_file = os.path.join(self.temp_dir, 'score_dist.png')
        
        interpretation = self.visualizer.create_visualization(
            data_file=self.test_file,
            viz_type="score_distribution",
            output_file=output_file
        )
        
        self.assertIsInstance(interpretation, str)
        self.assertTrue(len(interpretation) > 0)
    
    def test_create_visualization_predicted_score_boxplot(self):
        """Test creating predicted score boxplot visualization"""
        output_file = os.path.join(self.temp_dir, 'boxplot.png')
        
        interpretation = self.visualizer.create_visualization(
            data_file=self.test_file,
            viz_type="predicted_score_boxplot",
            output_file=output_file
        )
        
        self.assertIsInstance(interpretation, str)
        self.assertTrue(len(interpretation) > 0)
    
    def test_create_visualization_shared_genes_pathways(self):
        """Test creating shared genes pathways visualization"""
        output_file = os.path.join(self.temp_dir, 'shared.png')
        
        interpretation = self.visualizer.create_visualization(
            data_file=self.test_file,
            viz_type="shared_genes_pathways",
            output_file=output_file
        )
        
        self.assertIsInstance(interpretation, str)
        self.assertTrue(len(interpretation) > 0)
    
    def test_create_visualization_network_centrality(self):
        """Test creating network centrality visualization"""
        output_file = os.path.join(self.temp_dir, 'centrality.png')
        
        interpretation = self.visualizer.create_visualization(
            data_file=self.test_file,
            viz_type="network_centrality",
            output_file=output_file
        )
        
        self.assertIsInstance(interpretation, str)
        self.assertTrue(len(interpretation) > 0)
    
    def test_create_visualization_key_genes(self):
        """Test creating key genes visualization"""
        output_file = os.path.join(self.temp_dir, 'key_genes.png')
        
        interpretation = self.visualizer.create_visualization(
            data_file=self.test_file,
            viz_type="key_genes_distribution",
            output_file=output_file
        )
        
        self.assertIsInstance(interpretation, str)
        self.assertTrue(len(interpretation) > 0)
    
    def test_create_visualization_invalid_type(self):
        """Test creating visualization with invalid type"""
        output_file = os.path.join(self.temp_dir, 'invalid.png')
        
        with self.assertRaises(ValueError):
            self.visualizer.create_visualization(
                data_file=self.test_file,
                viz_type="invalid_type",
                output_file=output_file
            )
    
    def test_create_visualization_file_not_found(self):
        """Test creating visualization with non-existent file"""
        data_file = os.path.join(self.temp_dir, 'nonexistent.xlsx')
        output_file = os.path.join(self.temp_dir, 'test_viz.png')
        
        with self.assertRaises(FileNotFoundError):
            self.visualizer.create_visualization(
                data_file=data_file,
                viz_type='score_distribution',
                output_file=output_file
            )
    
    def test_create_visualization_custom_sheet_names(self):
        """Test creating visualization with custom sheet names"""
        data_file = os.path.join(self.temp_dir, 'test_data.xlsx')
        output_file = os.path.join(self.temp_dir, 'test_viz.png')
        
        # Create test data file with custom sheet name
        with pd.ExcelWriter(data_file) as writer:
            self.test_df.to_excel(writer, sheet_name='CustomSheet', index=False)
        
        interpretation = self.visualizer.create_visualization(
            data_file=data_file,
            viz_type='score_distribution',
            output_file=output_file,
            sheet_names=['CustomSheet', 'CustomSheet']
        )
        
        # Check if file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check if interpretation contains expected text
        self.assertIn('score distribution', interpretation.lower())
    
    def test_generate_report(self):
        """Test generating complete report"""
        output_file = os.path.join(self.temp_dir, 'report.html')
        
        result = self.visualizer.generate_report(
            data_file=self.test_file,
            output_file=output_file
        )
        
        self.assertIsInstance(result, str)
        self.assertTrue("Report generated" in result)
        self.assertTrue(os.path.exists(output_file))
    
    def test_generate_report_custom_title(self):
        """Test generating report with custom title"""
        output_file = os.path.join(self.temp_dir, 'custom_report.html')
        
        result = self.visualizer.generate_report(
            data_file=self.test_file,
            output_file=output_file,
            title="Custom Report Title"
        )
        
        self.assertIsInstance(result, str)
        self.assertTrue("Report generated" in result)
        self.assertTrue(os.path.exists(output_file))
    
    def test_generate_report_custom_sheets(self):
        """Test generating report with custom sheet names"""
        # Create test file with custom sheet names
        custom_file = os.path.join(self.temp_dir, 'custom_data.xlsx')
        df = pd.DataFrame(self.test_data)
        filtered_df = df[df['score'] > 0.5]
        
        with pd.ExcelWriter(custom_file) as writer:
            df.to_excel(writer, sheet_name='all_drugs', index=False)
            filtered_df.to_excel(writer, sheet_name='top_drugs', index=False)
        
        output_file = os.path.join(self.temp_dir, 'custom_sheets_report.html')
        
        result = self.visualizer.generate_report(
            data_file=custom_file,
            output_file=output_file,
            sheet_names=("all_drugs", "top_drugs")
        )
        
        self.assertIsInstance(result, str)
        self.assertTrue("Report generated" in result)
        self.assertTrue(os.path.exists(output_file))
    
    def test_plot_prompt(self):
        """Test prompt generation"""
        output_file = os.path.join(self.temp_dir, 'prompt.txt')
        
        df = pd.read_excel(self.test_file, sheet_name='annotated_drugs')
        filtered_df = pd.read_excel(self.test_file, sheet_name='filtered_drugs')
        
        interpretation = self.visualizer.plot_prompt(df, filtered_df, output_file)
        
        self.assertIsInstance(interpretation, str)
        self.assertTrue(len(interpretation) > 0)
        self.assertTrue(os.path.exists(output_file))
    
    def test_render_table(self):
        """Test table rendering"""
        df = pd.read_excel(self.test_file, sheet_name='annotated_drugs')
        
        table_html = Visualizer.render_table(df, "test_table")
        
        self.assertIsInstance(table_html, str)
        self.assertIn("test_table", table_html)
        self.assertIn("table", table_html)
    
    def test_visualization_with_empty_dataframe(self):
        """Test visualization with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        output_file = os.path.join(self.temp_dir, 'empty_viz.png')
        
        # Should handle empty DataFrame gracefully
        with self.assertRaises(ValueError):
            self.visualizer.plot_score_distribution(
                empty_df, empty_df, output_file
            )
    
    def test_visualization_with_missing_columns(self):
        """Test visualization with missing columns"""
        # Create data with missing columns
        minimal_data = {
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2'],
            'score': [0.9, 0.7]
        }
        
        minimal_file = os.path.join(self.temp_dir, 'minimal_data.xlsx')
        df = pd.DataFrame(minimal_data)
        filtered_df = df.copy()
        
        with pd.ExcelWriter(minimal_file) as writer:
            df.to_excel(writer, sheet_name='annotated_drugs', index=False)
            filtered_df.to_excel(writer, sheet_name='filtered_drugs', index=False)
        
        output_file = os.path.join(self.temp_dir, 'minimal.png')
        
        # This should work with minimal data
        interpretation = self.visualizer.create_visualization(
            data_file=minimal_file,
            viz_type="score_distribution",
            output_file=output_file
        )
        
        self.assertIsInstance(interpretation, str)
    
    def test_embed_images_option(self):
        """Test embed_images option"""
        visualizer_no_embed = Visualizer(
            disease_id="MESH:D001",
            disease_name="Test Disease",
            embed_images=False
        )
        
        self.assertFalse(visualizer_no_embed.embed_images)
        
        output_file = os.path.join(self.temp_dir, 'no_embed_report.html')
        
        result = visualizer_no_embed.generate_report(
            data_file=self.test_file,
            output_file=output_file
        )
        
        self.assertIsInstance(result, str)
        self.assertTrue("Report generated" in result)


if __name__ == '__main__':
    unittest.main() 