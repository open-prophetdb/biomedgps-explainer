#!/usr/bin/env python3
"""
Model类的测试文件
"""

import unittest
import tempfile
import os
import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pandas as pd
import numpy as np
import tarfile
import gzip

# 添加项目根目录到路径
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from drugs4disease.model import Model


class TestModel(unittest.TestCase):
    """Model类的测试用例"""
    
    def setUp(self):
        """测试前的设置"""
        self.project_name = "test-project"
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock model data
        self.mock_model_data = {
            'run_id': 'test_run_123',
            'run_name': 'test_run_name',
            'artifacts': [
                {
                    'name': 'model_files',
                    'type': 'model',
                    'state': 'logged'
                }
            ]
        }

    def tearDown(self):
        """测试后的清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_init(self, mock_api, mock_login):
        """测试Model类初始化"""
        mock_api_instance = Mock()
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        
        self.assertEqual(model.project_name, self.project_name)
        self.assertEqual(model.api, mock_api_instance)
        mock_login.assert_called_once()
        mock_api.assert_called_once()
        
        # Check if model root directory is created
        expected_model_root = os.path.expanduser("~/.biomedgps-explainer/models")
        self.assertTrue(os.path.exists(expected_model_root))
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_get_all_models(self, mock_api, mock_login):
        """测试获取所有模型信息"""
        mock_run = Mock()
        mock_run.id = 'test_run_123'
        mock_run.name = 'test_run_name'
        mock_run.display_name = 'Test Run'
        mock_run.state = 'finished'
        mock_run.created_at = '2024-01-01'
        mock_run.config = {}
        mock_run.summary = {}
        mock_run.logged_artifacts.return_value = []
        
        mock_api_instance = Mock()
        mock_api_instance.runs.return_value = [mock_run]
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        models = model.get_all_models()
        
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['run_id'], 'test_run_123')
        self.assertEqual(models[0]['run_name'], 'test_run_name')

    @patch('wandb.login')
    @patch('wandb.Api')
    def test_get_all_models_empty(self, mock_api, mock_login):
        """Test getting all models when project is empty"""
        mock_api_instance = Mock()
        mock_api_instance.runs.return_value = []
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        models = model.get_all_models()
        
        self.assertEqual(len(models), 0)

    @patch('wandb.login')
    @patch('wandb.Api')
    def test_get_all_models_error(self, mock_api, mock_login):
        """Test getting all models with API error"""
        mock_api_instance = Mock()
        mock_api_instance.runs.side_effect = Exception("API Error")
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        with self.assertRaises(Exception):
            model.get_all_models()
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_download_model_success(self, mock_api, mock_login):
        """Test successful model download"""
        mock_run = Mock()
        mock_run.id = 'test_run_123'
        mock_run.name = 'test_run_name'
        mock_run.logged_artifacts.return_value = []
        
        mock_api_instance = Mock()
        mock_api_instance.run.return_value = mock_run
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        
        # Mock the model directory check
        with patch.object(model, '_check_model_dir', return_value=True):
            model_dir = model.download_model("test_run_123")
        
        self.assertIsNotNone(model_dir)
        self.assertTrue(model_dir.endswith('test_run_name'))
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_download_model_existing_valid_dir(self, mock_api, mock_login):
        """测试下载模型时目录已存在且有效"""
        mock_run = Mock()
        mock_run.id = 'run1'
        mock_run.name = 'run1'
        mock_run.logged_artifacts.return_value = []
        
        mock_api_instance = Mock()
        mock_api_instance.run.return_value = mock_run
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        
        # Mock that the model directory exists and is valid
        with patch.object(model, '_check_model_dir', return_value=True):
            model_dir = model.download_model("run1")
        
        # Should return the existing directory path
        expected_dir = os.path.join(model.model_root_dir, "run1")
        self.assertEqual(model_dir, expected_dir)
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_download_model_not_found(self, mock_api, mock_login):
        mock_run = Mock()
        mock_run.name = 'non_existent_run'
        mock_api_instance = Mock()
        mock_api_instance.run.return_value = mock_run
        mock_api.return_value = mock_api_instance
        model = Model(self.project_name)
        with patch.object(model, '_check_model_dir', return_value=False):
            with self.assertRaises(Exception):
                model.download_model('non_existent_run')

    @patch('wandb.login')
    @patch('wandb.Api')
    def test_download_model_no_artifacts(self, mock_api, mock_login):
        mock_run = Mock()
        mock_run.name = 'test_run_name'
        mock_run.logged_artifacts.return_value = []
        mock_api_instance = Mock()
        mock_api_instance.run.return_value = mock_run
        mock_api.return_value = mock_api_instance
        model = Model(self.project_name)
        with patch.object(model, '_check_model_dir', return_value=False):
            model_dir = model.download_model('test_run_name')
            self.assertIsNotNone(model_dir)
            self.assertTrue(model_dir.endswith('test_run_name'))
    
    def test_validate_npy_file(self):
        """测试numpy文件验证"""
        model = Model("test")
        
        # 创建临时numpy文件
        temp_file = os.path.join(self.temp_dir, "test.npy")
        np.save(temp_file, np.array([1, 2, 3]))
        
        self.assertTrue(model._validate_npy_file(Path(temp_file)))
        
        # 测试无效文件
        invalid_file = os.path.join(self.temp_dir, "invalid.txt")
        with open(invalid_file, 'w') as f:
            f.write("not a numpy file")
        
        self.assertFalse(model._validate_npy_file(Path(invalid_file)))
    
    def test_validate_tsv_file(self):
        """测试tsv文件验证"""
        model = Model("test")
        
        # 创建有效的tsv文件
        temp_file = os.path.join(self.temp_dir, "test.tsv")
        df = pd.DataFrame([[1, "a"], [2, "b"]])
        df.to_csv(temp_file, sep="\t", header=False, index=False)
        
        self.assertTrue(model._validate_tsv_file(Path(temp_file), 2))
        self.assertFalse(model._validate_tsv_file(Path(temp_file), 3))
        
        # 测试不存在的文件
        self.assertFalse(model._validate_tsv_file(Path("nonexistent.tsv"), 2))
    
    def test_find_file(self):
        """测试文件查找"""
        model = Model("test")
        
        # 创建测试文件
        test_file = os.path.join(self.temp_dir, "entity_embeddings.npy")
        with open(test_file, 'w') as f:
            f.write("test")
        
        found_file = model._find_file(Path(self.temp_dir), "entity_embeddings")
        self.assertEqual(found_file, Path(test_file))
        
        # 测试找不到文件
        not_found = model._find_file(Path(self.temp_dir), "nonexistent")
        self.assertIsNone(not_found)
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_convert_model_files(self, mock_api, mock_login):
        """测试模型文件转换"""
        mock_api_instance = Mock()
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        model_dir = os.path.join(self.temp_dir, "test_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Mock the expected files
        with patch.object(model, '_get_expected_files') as mock_get_files:
            mock_get_files.return_value = {
                'entity_embeddings': os.path.join(model_dir, 'entity_embeddings.npy'),
                'entity_metadata': os.path.join(model_dir, 'entity_metadata.tar.gz'),
                'relation_embeddings': os.path.join(model_dir, 'relation_embeddings.npy'),
                'entity_id_map': os.path.join(model_dir, 'entity_id_map.tar.gz'),
                'relation_type_id_map': os.path.join(model_dir, 'relation_type_id_map.tar.gz'),
                'knowledge_graph': os.path.join(model_dir, 'knowledge_graph.tar.gz')
            }
            
            with patch.object(model, '_convert_entity_embeddings') as mock_convert_entity, \
                 patch.object(model, '_convert_relation_embeddings') as mock_convert_relation, \
                 patch.object(model, '_extract_tar_gz') as mock_extract:
                
                mock_convert_entity.return_value = os.path.join(model_dir, 'converted_entity.tsv')
                mock_convert_relation.return_value = os.path.join(model_dir, 'converted_relation.tsv')
                mock_extract.return_value = os.path.join(model_dir, 'extracted.tsv')
                
                result = model.convert_model_files(model_dir)
                
                self.assertIn('annotated_entities', result)
                self.assertIn('entity_embeddings', result)
                self.assertIn('relation_embeddings', result)
                self.assertIn('knowledge_graph', result)
                self.assertIn('model_dir', result)
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_load_model_config(self, mock_api, mock_login):
        """测试加载模型配置"""
        mock_api_instance = Mock()
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        model_dir = os.path.join(self.temp_dir, "test_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create mock config file
        config_data = {
            'model_name': 'TransE_l2',
            'gamma': 12.0,
            'embedding_dim': 100
        }
        config_file = os.path.join(model_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = model.load_model_config(model_dir)
        self.assertEqual(config['model_name'], 'TransE_l2')
        self.assertEqual(config['gamma'], 12.0)
        self.assertEqual(config['embedding_dim'], 100)
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_download_and_convert(self, mock_api, mock_login):
        """测试下载并转换模型"""
        mock_api_instance = Mock()
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        
        with patch.object(model, 'download_model') as mock_download, \
             patch.object(model, 'convert_model_files') as mock_convert:
            
            mock_download.return_value = os.path.join(self.temp_dir, "test_model")
            mock_convert.return_value = {
                'annotated_entities': 'entity.tsv',
                'entity_embeddings': 'entity_emb.tsv',
                'relation_embeddings': 'relation_emb.tsv',
                'knowledge_graph': 'kg.tsv',
                'model_dir': 'model_dir'
            }
            
            result = model.download_and_convert("run1")
            
            mock_download.assert_called_once_with("run1")
            mock_convert.assert_called_once()
            
            self.assertIn('annotated_entities', result)
            self.assertIn('entity_embeddings', result)
            self.assertIn('relation_embeddings', result)
            self.assertIn('knowledge_graph', result)
            self.assertIn('model_dir', result)
    
    def test_extract_tar_gz(self):
        """Test tar.gz file extraction"""
        model = Model(self.project_name)
        
        # Create a real tar.gz file for testing
        tar_file = os.path.join(self.temp_dir, "test.tar.gz")
        test_content = "test file content"
        
        with tarfile.open(tar_file, "w:gz") as tar:
            # Create a temporary file with content
            temp_file = os.path.join(self.temp_dir, "temp.txt")
            with open(temp_file, 'w') as f:
                f.write(test_content)
            
            # Add it to the tar archive
            tar.add(temp_file, arcname="test_file.txt")
        
        # Test extraction
        result = model._extract_tar_gz(Path(tar_file), Path(self.temp_dir))
        
        # Check if extraction was successful
        extracted_file = os.path.join(self.temp_dir, "test_file.txt")
        self.assertTrue(os.path.exists(extracted_file))
        
        # Check content
        with open(extracted_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, test_content)

    def test_extract_tar_gz_invalid_file(self):
        """Test extracting invalid tar.gz file"""
        model = Model(self.project_name)
        
        # Create an invalid file
        invalid_file = os.path.join(self.temp_dir, "invalid.tar.gz")
        with open(invalid_file, 'w') as f:
            f.write("not a tar file")
        
        with self.assertRaises(Exception):
            model._extract_tar_gz(Path(invalid_file), Path(self.temp_dir))

    def test_extract_tar_gz_file_not_found(self):
        """Test extracting non-existent tar.gz file"""
        model = Model(self.project_name)
        
        non_existent_file = os.path.join(self.temp_dir, "nonexistent.tar.gz")
        
        with self.assertRaises(FileNotFoundError):
            model._extract_tar_gz(Path(non_existent_file), Path(self.temp_dir))

    def test_check_model_dir(self):
        """测试检查模型目录"""
        model = Model("test")
        
        # Test with non-existent directory
        self.assertFalse(model._check_model_dir("non_existent_dir"))
        
        # Test with existing directory but missing files
        with patch.object(model, '_get_expected_files') as mock_get_files:
            mock_get_files.return_value = {
                'entity_embeddings': None,
                'entity_metadata': None,
                'relation_embeddings': None,
                'entity_id_map': None,
                'relation_type_id_map': None,
                'knowledge_graph': None
            }
            self.assertFalse(model._check_model_dir(self.temp_dir))
        
        # Test with all files present
        with patch.object(model, '_get_expected_files') as mock_get_files:
            mock_get_files.return_value = {
                'entity_emb': 'existing_file.npy',
                'entity_metadata': 'existing_file.tsv'
            }
            self.assertTrue(model._check_model_dir(self.temp_dir))


if __name__ == '__main__':
    unittest.main() 