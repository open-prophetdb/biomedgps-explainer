import unittest
import tempfile
import os
import shutil
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
from drugs4disease.full_pipeline import main


class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data for testing
        self.create_mock_data()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_data(self):
        """Create mock data files for testing"""
        # Mock entity file
        entity_data = {
            'id': ['MESH:D001', 'MESH:C001', 'MESH:C002'],
            'name': ['Test Disease', 'Drug1', 'Drug2'],
            'label': ['Disease', 'Compound', 'Compound']
        }
        self.entity_file = os.path.join(self.temp_dir, 'annotated_entities.tsv')
        pd.DataFrame(entity_data).to_csv(self.entity_file, sep='\t', index=False)
        
        # Mock knowledge graph
        kg_data = {
            'source_id': ['MESH:C001', 'MESH:C002'],
            'source_type': ['Compound', 'Compound'],
            'source_name': ['Drug1', 'Drug2'],
            'target_id': ['MESH:D001', 'MESH:D001'],
            'target_type': ['Disease', 'Disease'],
            'target_name': ['Test Disease', 'Test Disease'],
            'relation_type': ['GNBR::T::Compound:Disease', 'GNBR::T::Compound:Disease']
        }
        self.knowledge_graph = os.path.join(self.temp_dir, 'knowledge_graph.tsv')
        pd.DataFrame(kg_data).to_csv(self.knowledge_graph, sep='\t', index=False)
        
        # Mock entity embeddings
        embedding_data = {
            'entity_id': ['MESH:D001', 'MESH:C001', 'MESH:C002'],
            'entity_type': ['Disease', 'Compound', 'Compound'],
            'embedding': ['0.1|0.2|0.3', '0.4|0.5|0.6', '0.7|0.8|0.9']
        }
        self.entity_embeddings = os.path.join(self.temp_dir, 'entity_embeddings.tsv')
        pd.DataFrame(embedding_data).to_csv(self.entity_embeddings, sep='\t', index=False)
        
        # Mock relation embeddings
        relation_data = {
            'relation_type': ['GNBR::T::Compound:Disease'],
            'embedding': ['0.1|0.2|0.3']
        }
        self.relation_embeddings = os.path.join(self.temp_dir, 'relation_type_embeddings.tsv')
        pd.DataFrame(relation_data).to_csv(self.relation_embeddings, sep='\t', index=False)
    
    @patch('drugs4disease.full_pipeline.DrugDiseaseCore')
    @patch('drugs4disease.full_pipeline.DrugFilter')
    @patch('drugs4disease.full_pipeline.Model')
    @patch('drugs4disease.full_pipeline.validate_model_files')
    def test_main_successful_pipeline(self, mock_validate, mock_model_class, mock_filter_class, mock_core_class):
        """Test successful execution of the full pipeline"""
        # Mock the core components
        mock_core = Mock()
        mock_filter = Mock()
        mock_model = Mock()
        
        mock_core_class.return_value = mock_core
        mock_filter_class.return_value = mock_filter
        mock_model_class.return_value = mock_model
        
        # Mock model download and conversion
        mock_model.download_and_convert.return_value = {
            'annotated_entities': self.entity_file,
            'knowledge_graph': self.knowledge_graph,
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'model_dir': self.temp_dir
        }
        
        # Mock model config
        mock_model.load_model_config.return_value = {
            'model_name': 'TransE_l2',
            'gamma': 12.0
        }
        
        # Mock file validation
        mock_validate.return_value = True
        
        # Mock the full pipeline execution
        mock_core.run_full_pipeline.return_value = None
        
        # Create mock output files
        annotated_file = os.path.join(self.temp_dir, 'annotated_drugs.xlsx')
        test_data = pd.DataFrame({
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2'],
            'score': [0.8, 0.6]
        })
        with pd.ExcelWriter(annotated_file) as writer:
            test_data.to_excel(writer, sheet_name='annotated_drugs', index=False)
        
        # Mock file existence check
        with patch('os.path.exists', return_value=True):
            # Run the pipeline
            main(
                disease_id='MESH:D001',
                model_run_id='test_run',
                output_dir=self.temp_dir,
                filter_expression='score >= 0.5',
                top_n_diseases=10,
                threshold=0.5,
                relation_type='GNBR::T::Compound:Disease',
                top_n_drugs=100
            )
        
        # Verify that all components were called
        mock_core.run_full_pipeline.assert_called_once()
        mock_filter.filter_drugs.assert_called_once()
    
    @patch('drugs4disease.full_pipeline.DrugDiseaseCore')
    @patch('drugs4disease.full_pipeline.DrugFilter')
    @patch('drugs4disease.full_pipeline.Model')
    @patch('drugs4disease.full_pipeline.validate_model_files')
    def test_main_with_default_parameters(self, mock_validate, mock_model_class, mock_filter_class, mock_core_class):
        """Test pipeline execution with default parameters"""
        # Mock the core components
        mock_core = Mock()
        mock_filter = Mock()
        mock_model = Mock()
        
        mock_core_class.return_value = mock_core
        mock_filter_class.return_value = mock_filter
        mock_model_class.return_value = mock_model
        
        # Mock model download and conversion
        mock_model.download_and_convert.return_value = {
            'annotated_entities': self.entity_file,
            'knowledge_graph': self.knowledge_graph,
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'model_dir': self.temp_dir
        }
        
        # Mock model config
        mock_model.load_model_config.return_value = {
            'model_name': 'TransE_l2',
            'gamma': 12.0
        }
        
        # Mock file validation
        mock_validate.return_value = True
        
        # Mock the full pipeline execution
        mock_core.run_full_pipeline.return_value = None
        
        # Create mock output files
        annotated_file = os.path.join(self.temp_dir, 'annotated_drugs.xlsx')
        test_data = pd.DataFrame({
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2'],
            'score': [0.8, 0.6]
        })
        with pd.ExcelWriter(annotated_file) as writer:
            test_data.to_excel(writer, sheet_name='annotated_drugs', index=False)
        
        # Mock file existence check
        with patch('os.path.exists', return_value=True):
            # Run the pipeline with minimal parameters
            main(
                disease_id='MESH:D001',
                output_dir=self.temp_dir
            )
        
        # Verify that the pipeline was called with default parameters
        call_args = mock_core.run_full_pipeline.call_args
        self.assertEqual(call_args[1]['top_n_diseases'], 50)
        self.assertEqual(call_args[1]['threshold'], 0.5)
        self.assertEqual(call_args[1]['relation_type'], 'GNBR::T::Compound:Disease')
        self.assertEqual(call_args[1]['top_n_drugs'], 100)
    
    @patch('drugs4disease.full_pipeline.DrugDiseaseCore')
    @patch('drugs4disease.full_pipeline.DrugFilter')
    @patch('drugs4disease.full_pipeline.Model')
    @patch('drugs4disease.full_pipeline.validate_model_files')
    def test_main_validation_failure(self, mock_validate, mock_model_class, mock_filter_class, mock_core_class):
        """Test pipeline execution when file validation fails"""
        # Mock the core components
        mock_core = Mock()
        mock_filter = Mock()
        mock_model = Mock()
        
        mock_core_class.return_value = mock_core
        mock_filter_class.return_value = mock_filter
        mock_model_class.return_value = mock_model
        
        # Mock model download and conversion
        mock_model.download_and_convert.return_value = {
            'annotated_entities': self.entity_file,
            'knowledge_graph': self.knowledge_graph,
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'model_dir': self.temp_dir
        }
        
        # Mock model config
        mock_model.load_model_config.return_value = {
            'model_name': 'TransE_l2',
            'gamma': 12.0
        }
        
        # Mock file validation failure
        mock_validate.return_value = False
        
        # Mock sys.exit
        with patch('sys.exit') as mock_exit:
            # Run the pipeline
            main(
                disease_id='MESH:D001',
                output_dir=self.temp_dir
            )
            
            # Verify that sys.exit was called
            mock_exit.assert_called_once_with(1)
    
    @patch('drugs4disease.full_pipeline.DrugDiseaseCore')
    @patch('drugs4disease.full_pipeline.DrugFilter')
    @patch('drugs4disease.full_pipeline.Model')
    @patch('drugs4disease.full_pipeline.validate_model_files')
    def test_main_missing_model_config(self, mock_validate, mock_model_class, mock_filter_class, mock_core_class):
        """Test pipeline execution when model config is missing required fields"""
        # Mock the core components
        mock_core = Mock()
        mock_filter = Mock()
        mock_model = Mock()
        
        mock_core_class.return_value = mock_core
        mock_filter_class.return_value = mock_filter
        mock_model_class.return_value = mock_model
        
        # Mock model download and conversion
        mock_model.download_and_convert.return_value = {
            'annotated_entities': self.entity_file,
            'knowledge_graph': self.knowledge_graph,
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'model_dir': self.temp_dir
        }
        
        # Mock model config with missing fields
        mock_model.load_model_config.return_value = {
            'model_name': None,  # Missing model name
            'gamma': 12.0
        }
        
        # Mock file validation
        mock_validate.return_value = True
        
        # Should raise AssertionError for missing model name
        with self.assertRaises(AssertionError):
            main(
                disease_id='MESH:D001',
                output_dir=self.temp_dir
            )
    
    @patch('drugs4disease.full_pipeline.DrugDiseaseCore')
    @patch('drugs4disease.full_pipeline.DrugFilter')
    @patch('drugs4disease.full_pipeline.Model')
    @patch('drugs4disease.full_pipeline.validate_model_files')
    def test_main_missing_gamma(self, mock_validate, mock_model_class, mock_filter_class, mock_core_class):
        """Test pipeline execution when gamma is missing from model config"""
        # Mock the core components
        mock_core = Mock()
        mock_filter = Mock()
        mock_model = Mock()
        
        mock_core_class.return_value = mock_core
        mock_filter_class.return_value = mock_filter
        mock_model_class.return_value = mock_model
        
        # Mock model download and conversion
        mock_model.download_and_convert.return_value = {
            'annotated_entities': self.entity_file,
            'knowledge_graph': self.knowledge_graph,
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'model_dir': self.temp_dir
        }
        
        # Mock model config with missing gamma
        mock_model.load_model_config.return_value = {
            'model_name': 'TransE_l2',
            'gamma': None  # Missing gamma
        }
        
        # Mock file validation
        mock_validate.return_value = True
        
        # Should raise AssertionError for missing gamma
        with self.assertRaises(AssertionError):
            main(
                disease_id='MESH:D001',
                output_dir=self.temp_dir
            )
    
    @patch('drugs4disease.full_pipeline.DrugDiseaseCore')
    @patch('drugs4disease.full_pipeline.DrugFilter')
    @patch('drugs4disease.full_pipeline.Model')
    @patch('drugs4disease.full_pipeline.validate_model_files')
    def test_main_custom_filter_expression(self, mock_validate, mock_model_class, mock_filter_class, mock_core_class):
        """Test pipeline execution with custom filter expression"""
        # Mock the core components
        mock_core = Mock()
        mock_filter = Mock()
        mock_model = Mock()
        
        mock_core_class.return_value = mock_core
        mock_filter_class.return_value = mock_filter
        mock_model_class.return_value = mock_model
        
        # Mock model download and conversion
        mock_model.download_and_convert.return_value = {
            'annotated_entities': self.entity_file,
            'knowledge_graph': self.knowledge_graph,
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'model_dir': self.temp_dir
        }
        
        # Mock model config
        mock_model.load_model_config.return_value = {
            'model_name': 'TransE_l2',
            'gamma': 12.0
        }
        
        # Mock file validation
        mock_validate.return_value = True
        
        # Mock the full pipeline execution
        mock_core.run_full_pipeline.return_value = None
        
        # Create mock output files
        annotated_file = os.path.join(self.temp_dir, 'annotated_drugs.xlsx')
        test_data = pd.DataFrame({
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2'],
            'score': [0.8, 0.6]
        })
        with pd.ExcelWriter(annotated_file) as writer:
            test_data.to_excel(writer, sheet_name='annotated_drugs', index=False)
        
        # Mock file existence check
        with patch('os.path.exists', return_value=True):
            # Run the pipeline with custom filter expression
            custom_filter = 'score >= 0.7 && drug_name == "Drug1"'
            main(
                disease_id='MESH:D001',
                output_dir=self.temp_dir,
                filter_expression=custom_filter
            )
        
        # Verify that the filter was called with custom expression
        call_args = mock_filter.filter_drugs.call_args
        self.assertEqual(call_args[1]['expression'], custom_filter)
    
    @patch('drugs4disease.full_pipeline.DrugDiseaseCore')
    @patch('drugs4disease.full_pipeline.DrugFilter')
    @patch('drugs4disease.full_pipeline.Model')
    @patch('drugs4disease.full_pipeline.validate_model_files')
    def test_main_visualization_report_generation(self, mock_validate, mock_model_class, mock_filter_class, mock_core_class):
        """Test pipeline execution with visualization report generation"""
        # Mock the core components
        mock_core = Mock()
        mock_filter = Mock()
        mock_model = Mock()
        
        mock_core_class.return_value = mock_core
        mock_filter_class.return_value = mock_filter
        mock_model_class.return_value = mock_model
        
        # Mock model download and conversion
        mock_model.download_and_convert.return_value = {
            'annotated_entities': self.entity_file,
            'knowledge_graph': self.knowledge_graph,
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'model_dir': self.temp_dir
        }
        
        # Mock model config
        mock_model.load_model_config.return_value = {
            'model_name': 'TransE_l2',
            'gamma': 12.0
        }
        
        # Mock file validation
        mock_validate.return_value = True
        
        # Mock the full pipeline execution
        mock_core.run_full_pipeline.return_value = None
        
        # Mock disease name retrieval
        mock_core.get_disease_name.return_value = "Test Disease"
        
        # Create mock output files
        annotated_file = os.path.join(self.temp_dir, 'annotated_drugs.xlsx')
        filtered_file = os.path.join(self.temp_dir, 'filtered_drugs.xlsx')
        
        test_data = pd.DataFrame({
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2'],
            'score': [0.8, 0.6]
        })
        
        with pd.ExcelWriter(annotated_file) as writer:
            test_data.to_excel(writer, sheet_name='annotated_drugs', index=False)
        
        with pd.ExcelWriter(filtered_file) as writer:
            test_data.to_excel(writer, sheet_name='annotated_drugs', index=False)
            test_data.to_excel(writer, sheet_name='filtered_drugs', index=False)
        
        # Mock file existence check
        with patch('os.path.exists', return_value=True):
            # Mock Visualizer
            with patch('drugs4disease.full_pipeline.Visualizer') as mock_visualizer_class:
                mock_visualizer = Mock()
                mock_visualizer_class.return_value = mock_visualizer
                mock_visualizer.generate_report.return_value = "Report generated successfully"
                
                # Run the pipeline
                main(
                    disease_id='MESH:D001',
                    output_dir=self.temp_dir
                )
                
                # Verify that visualization was attempted
                mock_visualizer.generate_report.assert_called_once()
    
    @patch('drugs4disease.full_pipeline.DrugDiseaseCore')
    @patch('drugs4disease.full_pipeline.DrugFilter')
    @patch('drugs4disease.full_pipeline.Model')
    @patch('drugs4disease.full_pipeline.validate_model_files')
    def test_main_exception_handling(self, mock_validate, mock_model_class, mock_filter_class, mock_core_class):
        """Test pipeline execution with exception handling"""
        # Mock the core components
        mock_core = Mock()
        mock_filter = Mock()
        mock_model = Mock()
        
        mock_core_class.return_value = mock_core
        mock_filter_class.return_value = mock_filter
        mock_model_class.return_value = mock_model
        
        # Mock model download and conversion
        mock_model.download_and_convert.return_value = {
            'annotated_entities': self.entity_file,
            'knowledge_graph': self.knowledge_graph,
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'model_dir': self.temp_dir
        }
        
        # Mock model config
        mock_model.load_model_config.return_value = {
            'model_name': 'TransE_l2',
            'gamma': 12.0
        }
        
        # Mock file validation
        mock_validate.return_value = True
        
        # Mock the full pipeline execution to raise an exception
        mock_core.run_full_pipeline.side_effect = Exception("Test error")
        
        # Mock traceback
        with patch('traceback.print_exc') as mock_traceback:
            # Run the pipeline
            main(
                disease_id='MESH:D001',
                output_dir=self.temp_dir
            )
            
            # Verify that traceback was called
            mock_traceback.assert_called_once()


if __name__ == '__main__':
    unittest.main() 