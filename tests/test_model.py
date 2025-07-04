#!/usr/bin/env python3
"""
Model类的测试文件
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

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
        
    def tearDown(self):
        """测试后的清理"""
        import shutil
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
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_get_all_models(self, mock_api, mock_login):
        """测试获取所有模型信息"""
        # 模拟API和运行
        mock_api_instance = Mock()
        mock_run1 = Mock()
        mock_run1.id = "run1"
        mock_run1.name = "test_run_1"
        mock_run1.state = "finished"
        mock_run1.created_at = "2023-01-01"
        mock_run1.updated_at = "2023-01-02"
        mock_run1.config = {"param1": "value1"}
        mock_run1.summary = {"metric1": 0.95}
        
        mock_artifact = Mock()
        mock_artifact.name = "test_artifact"
        mock_artifact.version = "v1"
        mock_artifact.type = "model"
        mock_artifact.description = "Test artifact"
        mock_artifact.metadata = {}
        mock_artifact.size = 1024
        mock_artifact.created_at = "2023-01-01"
        mock_artifact.updated_at = "2023-01-02"
        
        mock_run1.logged_artifacts.return_value = [mock_artifact]
        mock_api_instance.runs.return_value = [mock_run1]
        
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        models = model.get_all_models()
        
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['run_id'], "run1")
        self.assertEqual(models[0]['run_name'], "test_run_1")
        self.assertEqual(len(models[0]['artifacts']), 1)
        self.assertEqual(models[0]['artifacts'][0]['name'], "test_artifact")
    
    @patch('wandb.login')
    @patch('wandb.Api')
    def test_download_model(self, mock_api, mock_login):
        """测试下载模型"""
        # 模拟API和运行
        mock_api_instance = Mock()
        mock_run = Mock()
        mock_run.id = "run1"
        mock_run.name = "test_run"
        
        mock_artifact = Mock()
        mock_artifact.name = "test:artifact"
        mock_artifact.version = "v1"
        mock_artifact.type = "model"
        
        mock_run.logged_artifacts.return_value = [mock_artifact]
        mock_api_instance.run.return_value = mock_run
        
        mock_api.return_value = mock_api_instance
        
        model = Model(self.project_name)
        
        # 测试下载
        with patch('os.makedirs'), \
             patch('os.walk') as mock_walk, \
             patch('shutil.move'), \
             patch('shutil.rmtree'):
            
            mock_walk.return_value = [("temp_dir", [], ["test_file.npy"])]
            
            model_dir = model.download_model("run1", self.temp_dir)
            
            self.assertIn("test_run", model_dir)
            mock_artifact.download.assert_called_once()
    
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


if __name__ == '__main__':
    unittest.main() 