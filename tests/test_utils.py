import unittest
import tempfile
import os
import shutil
import zipfile
from unittest.mock import patch, mock_open, MagicMock
from drugs4disease.utils import (
    extract_zip_if_needed,
    extract_model_files_if_needed,
    get_default_model_dir,
    _find_project_root,
    _search_project_root_from_dir,
    get_model_file_paths,
    validate_model_files,
    init_logger
)
import pandas as pd


class TestUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_entity_data = {
            'id': ['MESH:D001', 'MESH:C001'],
            'label': ['Disease', 'Compound'],
            'name': ['Disease1', 'Drug1']
        }
        
        self.test_kg_data = {
            'source_id': ['MESH:C001'],
            'source_type': ['Compound'],
            'source_name': ['Drug1'],
            'target_id': ['MESH:D001'],
            'target_type': ['Disease'],
            'target_name': ['Disease1'],
            'relation_type': ['GNBR::T::Compound:Disease']
        }
        
        self.test_entity_embeddings_data = {
            'embedding_id': ['MESH:D001', 'MESH:C001'],
            'entity_id': ['MESH:D001', 'MESH:C001'],
            'entity_type': ['Disease', 'Compound'],
            'entity_name': ['Disease1', 'Drug1'],
            'embedding': ['0.1|0.2|0.3', '0.4|0.5|0.6']
        }
        
        self.test_relation_embeddings_data = {
            'id': ['GNBR::T::Compound:Disease'],
            'embedding': ['0.1|0.2|0.3']
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_zip_if_needed_with_zip_file(self):
        """Test ZIP file extraction when ZIP file exists"""
        # Create a mock ZIP file
        zip_path = os.path.join(self.temp_dir, "test.zip")
        extract_dir = os.path.join(self.temp_dir, "extract")
        
        # Create a simple ZIP file
        with zipfile.ZipFile(zip_path, 'w') as zip_ref:
            zip_ref.writestr("test.txt", "test content")
        
        result = extract_zip_if_needed(zip_path, extract_dir)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(extract_dir))
        self.assertTrue(os.path.exists(os.path.join(extract_dir, "test.txt")))
    
    def test_extract_zip_if_needed_without_zip_file(self):
        """Test ZIP file extraction when ZIP file doesn't exist"""
        zip_path = os.path.join(self.temp_dir, "nonexistent.zip")
        extract_dir = os.path.join(self.temp_dir, "extract")
        
        result = extract_zip_if_needed(zip_path, extract_dir)
        
        self.assertFalse(result)
    
    def test_extract_zip_if_needed_invalid_zip(self):
        """Test ZIP file extraction with invalid ZIP file"""
        # Create an invalid ZIP file
        zip_path = os.path.join(self.temp_dir, "invalid.zip")
        extract_dir = os.path.join(self.temp_dir, "extract")
        
        with open(zip_path, 'w') as f:
            f.write("not a zip file")
        
        result = extract_zip_if_needed(zip_path, extract_dir)
        
        self.assertFalse(result)
    
    def test_extract_model_files_if_needed_all_files_exist(self):
        """Test model file extraction when all files already exist"""
        model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create all required files
        required_files = [
            "annotated_entities.tsv",
            "knowledge_graph.tsv", 
            "entity_embeddings.tsv",
            "relation_type_embeddings.tsv"
        ]
        
        for file in required_files:
            with open(os.path.join(model_dir, file), 'w') as f:
                f.write("test content")
        
        result = extract_model_files_if_needed(model_dir)
        
        self.assertTrue(result)
    
    def test_extract_model_files_if_needed_with_zip_files(self):
        """Test model file extraction when ZIP files exist"""
        model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create ZIP files
        zip_files = [
            "annotated_entities.tsv.zip",
            "knowledge_graph.tsv.zip",
            "entity_embeddings.tsv.zip", 
            "relation_type_embeddings.tsv.zip"
        ]
        
        for zip_file in zip_files:
            zip_path = os.path.join(model_dir, zip_file)
            with zipfile.ZipFile(zip_path, 'w') as zip_ref:
                # Create the corresponding TSV file inside the ZIP
                tsv_name = zip_file.replace('.zip', '')
                zip_ref.writestr(tsv_name, "test content")
        
        result = extract_model_files_if_needed(model_dir)
        
        self.assertTrue(result)
        
        # Check if TSV files were extracted
        required_files = [
            "annotated_entities.tsv",
            "knowledge_graph.tsv", 
            "entity_embeddings.tsv",
            "relation_type_embeddings.tsv"
        ]
        
        for file in required_files:
            self.assertTrue(os.path.exists(os.path.join(model_dir, file)))
    
    def test_extract_model_files_if_needed_missing_files(self):
        """Test model file extraction when files are missing"""
        model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Don't create any files
        
        result = extract_model_files_if_needed(model_dir)
        
        self.assertFalse(result)
    
    def test_extract_model_files_if_needed_nonexistent_dir(self):
        """Test model file extraction with non-existent directory"""
        model_dir = os.path.join(self.temp_dir, "nonexistent")
        
        result = extract_model_files_if_needed(model_dir)
        
        self.assertFalse(result)
    
    def test_get_default_model_dir(self):
        """Test getting default model directory"""
        model_dir = get_default_model_dir()
        
        # Should return a path
        self.assertIsInstance(model_dir, str)
        self.assertTrue(len(model_dir) > 0)
    
    def test_search_project_root_from_dir_with_data_dir(self):
        """Test searching for project root with data directory"""
        # Create a mock project structure
        project_root = os.path.join(self.temp_dir, "project")
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        result = _search_project_root_from_dir(project_root)
        
        self.assertEqual(result, project_root)
    
    def test_search_project_root_from_dir_with_setup_py(self):
        """Test searching for project root with setup.py"""
        # Create a mock project structure
        project_root = os.path.join(self.temp_dir, "project")
        setup_py = os.path.join(project_root, "setup.py")
        os.makedirs(project_root, exist_ok=True)
        
        with open(setup_py, 'w') as f:
            f.write("setup content")
        
        result = _search_project_root_from_dir(project_root)
        
        self.assertEqual(result, project_root)
    
    def test_search_project_root_from_dir_with_pyproject_toml(self):
        """Test searching for project root with pyproject.toml"""
        # Create a mock project structure
        project_root = os.path.join(self.temp_dir, "project")
        pyproject_toml = os.path.join(project_root, "pyproject.toml")
        os.makedirs(project_root, exist_ok=True)
        
        with open(pyproject_toml, 'w') as f:
            f.write("project content")
        
        result = _search_project_root_from_dir(project_root)
        
        self.assertEqual(result, project_root)
    
    def test_search_project_root_from_dir_with_git(self):
        """Test searching for project root with .git directory"""
        # Create a mock project structure
        project_root = os.path.join(self.temp_dir, "project")
        git_dir = os.path.join(project_root, ".git")
        os.makedirs(git_dir, exist_ok=True)
        
        result = _search_project_root_from_dir(project_root)
        
        self.assertEqual(result, project_root)
    
    def test_search_project_root_from_dir_with_readme(self):
        """Test searching for project root with README.md"""
        # Create a mock project structure
        project_root = os.path.join(self.temp_dir, "project")
        readme = os.path.join(project_root, "README.md")
        os.makedirs(project_root, exist_ok=True)
        
        with open(readme, 'w') as f:
            f.write("readme content")
        
        result = _search_project_root_from_dir(project_root)
        
        self.assertEqual(result, project_root)
    
    def test_search_project_root_from_dir_no_indicators(self):
        """Test searching for project root with no indicators"""
        # Create a directory with no project indicators
        test_dir = os.path.join(self.temp_dir, "no_project")
        os.makedirs(test_dir, exist_ok=True)
        
        result = _search_project_root_from_dir(test_dir)
        
        self.assertIsNone(result)
    
    def test_find_project_root(self):
        """Test finding project root"""
        # Create a mock project structure
        project_root = os.path.join(self.temp_dir, "project")
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        with patch('drugs4disease.utils._search_project_root_from_dir') as mock_search:
            mock_search.return_value = project_root
            
            result = _find_project_root()
            
            self.assertEqual(result, project_root)
    
    def test_get_model_file_paths_all_specified(self):
        """Test getting model file paths when all are specified"""
        # Create test files
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        knowledge_graph = os.path.join(self.temp_dir, 'kg.tsv')
        entity_embeddings = os.path.join(self.temp_dir, 'entity_emb.tsv')
        relation_embeddings = os.path.join(self.temp_dir, 'relation_emb.tsv')
        
        # Create the files
        pd.DataFrame(self.test_entity_data).to_csv(entity_file, sep='\t', index=False)
        pd.DataFrame(self.test_kg_data).to_csv(knowledge_graph, sep='\t', index=False)
        pd.DataFrame(self.test_entity_embeddings_data).to_csv(entity_embeddings, sep='\t', index=False)
        pd.DataFrame(self.test_relation_embeddings_data).to_csv(relation_embeddings, sep='\t', index=False)
        
        result = get_model_file_paths(
            entity_file, knowledge_graph, entity_embeddings, relation_embeddings
        )
        
        expected_paths = (entity_file, knowledge_graph, entity_embeddings, relation_embeddings)
        self.assertEqual(result, expected_paths)
    
    def test_get_model_file_paths_none_specified(self):
        """Test getting model file paths when none are specified"""
        with patch('drugs4disease.utils.get_default_model_dir') as mock_default:
            mock_default.return_value = self.temp_dir
            
            # Create mock files in the default directory
            entity_file = os.path.join(self.temp_dir, 'annotated_entities.tsv')
            kg_file = os.path.join(self.temp_dir, 'knowledge_graph.tsv')
            entity_emb_file = os.path.join(self.temp_dir, 'entity_embeddings.tsv')
            relation_emb_file = os.path.join(self.temp_dir, 'relation_type_embeddings.tsv')
            
            for file_path in [entity_file, kg_file, entity_emb_file, relation_emb_file]:
                pd.DataFrame({'test': ['data']}).to_csv(file_path, sep='\t', index=False)
            
            # Mock the extraction function
            with patch('drugs4disease.utils.extract_model_files_if_needed', return_value=True):
                result = get_model_file_paths()
            
            # Should return the default file paths
            expected_paths = (
                entity_file,
                kg_file,
                entity_emb_file,
                relation_emb_file
            )
            self.assertEqual(result, expected_paths)
    
    def test_get_model_file_paths_partial_specified(self):
        """Test getting model file paths when only some are specified"""
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        
        with self.assertRaises(ValueError):
            get_model_file_paths(entity_file=entity_file)
    
    def test_get_model_file_paths_file_not_found(self):
        """Test getting model file paths when files don't exist"""
        entity_file = os.path.join(self.temp_dir, 'nonexistent.tsv')
        knowledge_graph = os.path.join(self.temp_dir, 'nonexistent2.tsv')
        entity_embeddings = os.path.join(self.temp_dir, 'nonexistent3.tsv')
        relation_embeddings = os.path.join(self.temp_dir, 'nonexistent4.tsv')
        
        with self.assertRaises(FileNotFoundError):
            get_model_file_paths(
                entity_file, knowledge_graph, entity_embeddings, relation_embeddings
            )
    
    def test_validate_model_files_valid(self):
        """Test validating valid model files"""
        # Create valid test files
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        knowledge_graph = os.path.join(self.temp_dir, 'kg.tsv')
        entity_embeddings = os.path.join(self.temp_dir, 'entity_emb.tsv')
        relation_embeddings = os.path.join(self.temp_dir, 'relation_emb.tsv')
        
        # Create files with correct format
        pd.DataFrame(self.test_entity_data).to_csv(entity_file, sep='\t', index=False)
        pd.DataFrame(self.test_kg_data).to_csv(knowledge_graph, sep='\t', index=False)
        pd.DataFrame(self.test_entity_embeddings_data).to_csv(entity_embeddings, sep='\t', index=False)
        pd.DataFrame(self.test_relation_embeddings_data).to_csv(relation_embeddings, sep='\t', index=False)
        
        result = validate_model_files(
            entity_file, knowledge_graph, entity_embeddings, relation_embeddings
        )
        
        self.assertTrue(result)
    
    def test_validate_model_files_invalid_format(self):
        """Test model file validation with invalid file format"""
        # Create files with invalid format
        entity_file = os.path.join(self.temp_dir, 'entity.txt')  # Wrong extension
        knowledge_graph = os.path.join(self.temp_dir, 'kg.tsv')
        entity_embeddings = os.path.join(self.temp_dir, 'entity_emb.tsv')
        relation_embeddings = os.path.join(self.temp_dir, 'relation_emb.tsv')
        
        for file_path in [entity_file, knowledge_graph, entity_embeddings, relation_embeddings]:
            with open(file_path, 'w') as f:
                f.write("test content")
        
        result = validate_model_files(
            entity_file, knowledge_graph, entity_embeddings, relation_embeddings
        )
        
        self.assertFalse(result)
    
    def test_validate_model_files_missing_columns(self):
        """Test model file validation with missing columns"""
        # Create files with missing required columns
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        knowledge_graph = os.path.join(self.temp_dir, 'kg.tsv')
        entity_embeddings = os.path.join(self.temp_dir, 'entity_emb.tsv')
        relation_embeddings = os.path.join(self.temp_dir, 'relation_emb.tsv')
        
        # Create files with missing columns
        pd.DataFrame({'id': ['test']}).to_csv(entity_file, sep='\t', index=False)
        pd.DataFrame({'source_id': ['test']}).to_csv(knowledge_graph, sep='\t', index=False)
        pd.DataFrame({'entity_id': ['test']}).to_csv(entity_embeddings, sep='\t', index=False)
        pd.DataFrame({'id': ['test']}).to_csv(relation_embeddings, sep='\t', index=False)
        
        result = validate_model_files(
            entity_file, knowledge_graph, entity_embeddings, relation_embeddings
        )
        
        self.assertFalse(result)
    
    def test_validate_model_files_file_not_found(self):
        """Test model file validation with non-existent files"""
        entity_file = os.path.join(self.temp_dir, 'nonexistent.tsv')
        knowledge_graph = os.path.join(self.temp_dir, 'nonexistent2.tsv')
        entity_embeddings = os.path.join(self.temp_dir, 'nonexistent3.tsv')
        relation_embeddings = os.path.join(self.temp_dir, 'nonexistent4.tsv')
        
        result = validate_model_files(
            entity_file, knowledge_graph, entity_embeddings, relation_embeddings
        )
        
        self.assertFalse(result)
    
    def test_extract_model_files_if_needed_valid_directory(self):
        """Test extracting model files when directory is valid"""
        # Create a valid model directory
        model_dir = os.path.join(self.temp_dir, 'valid_model')
        os.makedirs(model_dir, exist_ok=True)
        
        # Create required files
        required_files = [
            'annotated_entities.tsv',
            'knowledge_graph.tsv',
            'entity_embeddings.tsv',
            'relation_type_embeddings.tsv'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(model_dir, file_name)
            pd.DataFrame({'test': ['data']}).to_csv(file_path, sep='\t', index=False)
        
        result = extract_model_files_if_needed(model_dir)
        self.assertTrue(result)
    
    def test_extract_model_files_if_needed_invalid_directory(self):
        """Test extracting model files when directory is invalid"""
        # Create an invalid model directory (missing files)
        model_dir = os.path.join(self.temp_dir, 'invalid_model')
        os.makedirs(model_dir, exist_ok=True)
        
        result = extract_model_files_if_needed(model_dir)
        self.assertFalse(result)
    
    def test_extract_model_files_if_needed_nonexistent_directory(self):
        """Test extracting model files when directory doesn't exist"""
        model_dir = os.path.join(self.temp_dir, 'nonexistent')
        
        result = extract_model_files_if_needed(model_dir)
        self.assertFalse(result)
    
    def test_extract_zip_if_needed_valid_zip(self):
        """Test extracting valid ZIP file"""
        # Create a test ZIP file
        zip_path = os.path.join(self.temp_dir, 'test.zip')
        extract_dir = self.temp_dir
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr('test_file.txt', 'test content')
        
        result = extract_zip_if_needed(zip_path, extract_dir)
        self.assertTrue(result)
        
        # Check if file was extracted
        extracted_file = os.path.join(extract_dir, 'test_file.txt')
        self.assertTrue(os.path.exists(extracted_file))
    
    def test_extract_zip_if_needed_invalid_zip(self):
        """Test extracting invalid ZIP file"""
        # Create an invalid ZIP file
        zip_path = os.path.join(self.temp_dir, 'invalid.zip')
        extract_dir = self.temp_dir
        
        with open(zip_path, 'w') as f:
            f.write('not a zip file')
        
        result = extract_zip_if_needed(zip_path, extract_dir)
        self.assertFalse(result)
    
    def test_extract_zip_if_needed_nonexistent_zip(self):
        """Test extracting non-existent ZIP file"""
        zip_path = os.path.join(self.temp_dir, 'nonexistent.zip')
        extract_dir = self.temp_dir
        
        result = extract_zip_if_needed(zip_path, extract_dir)
        self.assertFalse(result)
    
    def test_get_model_file_paths_with_zip_files(self):
        """Test getting model file paths with ZIP files"""
        # Create test files
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        knowledge_graph = os.path.join(self.temp_dir, 'kg.tsv')
        entity_embeddings = os.path.join(self.temp_dir, 'entity_emb.tsv')
        relation_embeddings = os.path.join(self.temp_dir, 'relation_emb.tsv')
        
        # Create ZIP files instead of regular files
        zip_files = [
            (entity_file + '.zip', 'entity.tsv'),
            (knowledge_graph + '.zip', 'kg.tsv'),
            (entity_embeddings + '.zip', 'entity_emb.tsv'),
            (relation_embeddings + '.zip', 'relation_emb.tsv')
        ]
        
        for zip_path, file_name in zip_files:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.writestr(file_name, 'test content')
        
        # Mock the extraction function to return True
        with patch('drugs4disease.utils.extract_zip_if_needed', return_value=True):
            result = get_model_file_paths(
                entity_file, knowledge_graph, entity_embeddings, relation_embeddings
            )
        
        expected_paths = (entity_file, knowledge_graph, entity_embeddings, relation_embeddings)
        self.assertEqual(result, expected_paths)
    
    def test_validate_model_files_empty_files(self):
        """Test validating empty model files"""
        # Create empty files
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        knowledge_graph = os.path.join(self.temp_dir, 'kg.tsv')
        entity_embeddings = os.path.join(self.temp_dir, 'entity_emb.tsv')
        relation_embeddings = os.path.join(self.temp_dir, 'relation_emb.tsv')
        
        for file_path in [entity_file, knowledge_graph, entity_embeddings, relation_embeddings]:
            with open(file_path, 'w') as f:
                f.write('')  # Empty file
        
        result = validate_model_files(
            entity_file, knowledge_graph, entity_embeddings, relation_embeddings
        )
        
        self.assertFalse(result)
    
    def test_validate_model_files_corrupted_files(self):
        """Test validating corrupted model files"""
        # Create corrupted files
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        knowledge_graph = os.path.join(self.temp_dir, 'kg.tsv')
        entity_embeddings = os.path.join(self.temp_dir, 'entity_emb.tsv')
        relation_embeddings = os.path.join(self.temp_dir, 'relation_emb.tsv')
        
        for file_path in [entity_file, knowledge_graph, entity_embeddings, relation_embeddings]:
            with open(file_path, 'w') as f:
                f.write('corrupted\tdata\nwith\tinvalid\tformat')  # Corrupted TSV
        
        result = validate_model_files(
            entity_file, knowledge_graph, entity_embeddings, relation_embeddings
        )
        
        self.assertFalse(result)
    
    def test_init_logger_default(self):
        """Test logger initialization with default parameters"""
        logger = init_logger()
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, 20)  # INFO level
    
    def test_init_logger_custom_level(self):
        """Test logger initialization with custom level"""
        logger = init_logger(log_level=10)  # DEBUG level
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, 10)
    
    def test_init_logger_with_file(self):
        """Test logger initialization with log file"""
        log_file = os.path.join(self.temp_dir, "test.log")
        
        logger = init_logger(log_file=log_file)
        
        self.assertIsNotNone(logger)
        self.assertTrue(os.path.exists(log_file))
    
    def test_init_logger_with_name(self):
        """Test logger initialization with custom name"""
        logger = init_logger(logger_name="test_logger")
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger")


if __name__ == '__main__':
    unittest.main() 