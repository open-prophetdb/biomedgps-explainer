import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from drugs4disease.core import DrugDiseaseCore

class TestDrugDiseaseCore(unittest.TestCase):
    def setUp(self):
        self.core = DrugDiseaseCore()
        # 使用BioMedGPS数据文件路径
        self.data_dir = "data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV"
        self.entity_file = os.path.join(self.data_dir, "annotated_entities.tsv")
        self.knowledge_graph = os.path.join(self.data_dir, "knowledge_graph.tsv")
        self.entity_embeddings = os.path.join(self.data_dir, "entity_embeddings.tsv")
        self.relation_embeddings = os.path.join(self.data_dir, "relation_type_embeddings.tsv")
        
    def test_data_files_exist(self):
        """测试数据文件是否存在"""
        self.assertTrue(os.path.exists(self.entity_file), "annotated_entities.tsv not found")
        self.assertTrue(os.path.exists(self.knowledge_graph), "knowledge_graph.tsv not found")
        self.assertTrue(os.path.exists(self.entity_embeddings), "entity_embeddings.tsv not found")
        self.assertTrue(os.path.exists(self.relation_embeddings), "relation_type_embeddings.tsv not found")
    
    def test_entity_file_format(self):
        """测试实体文件格式"""
        if os.path.exists(self.entity_file):
            df = pd.read_csv(self.entity_file, sep="\t", nrows=5)
            required_cols = ["id", "name", "label"]
            for col in required_cols:
                self.assertIn(col, df.columns, f"Missing column: {col}")
    
    def test_knowledge_graph_format(self):
        """测试知识图谱文件格式"""
        if os.path.exists(self.knowledge_graph):
            df = pd.read_csv(self.knowledge_graph, sep="\t", nrows=5)
            required_cols = ["source_id", "source_type", "target_id", "target_type", "relation_type"]
            for col in required_cols:
                self.assertIn(col, df.columns, f"Missing column: {col}")
    
    def test_entity_embeddings_format(self):
        """测试实体嵌入文件格式"""
        if os.path.exists(self.entity_embeddings):
            df = pd.read_csv(self.entity_embeddings, sep="\t", nrows=5)
            required_cols = ["entity_id", "entity_type", "embedding"]
            for col in required_cols:
                self.assertIn(col, df.columns, f"Missing column: {col}")
    
    def test_relation_embeddings_format(self):
        """测试关系嵌入文件格式"""
        if os.path.exists(self.relation_embeddings):
            df = pd.read_csv(self.relation_embeddings, sep="\t", nrows=5)
            # BioMedGPS格式可能使用不同的列名
            self.assertTrue(
                "embedding" in df.columns and 
                ("id" in df.columns or "relation_type" in df.columns),
                "Missing required columns in relation embeddings"
            )
    
    def test_predict_drugs_with_sample_data(self):
        """测试药物预测功能（使用样本数据）"""
        if not all(os.path.exists(f) for f in [self.entity_file, self.knowledge_graph, 
                                              self.entity_embeddings, self.relation_embeddings]):
            self.skipTest("Data files not available")
        
        # 创建一个临时输出目录
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_predicted_drugs.xlsx")
            
            # 使用一个已知的疾病ID进行测试
            # 从实体文件中找到一个疾病ID
            entity_df = pd.read_csv(self.entity_file, sep="\t")
            disease_entities = entity_df[entity_df["label"] == "Disease"]
            
            if len(disease_entities) > 0:
                test_disease_id = disease_entities.iloc[0]["id"]
                
                try:
                    self.core.predict_drugs(
                        disease_id=test_disease_id,
                        entity_file=self.entity_file,
                        knowledge_graph=self.knowledge_graph,
                        entity_embeddings=self.entity_embeddings,
                        relation_embeddings=self.relation_embeddings,
                        model='TransE_l2',
                        top_n_diseases=10,
                        gamma=12.0,
                        threshold=0.5,
                        relation_type='DGIDB::OTHER::Gene:Compound',
                        output_file=output_file
                    )
                    
                    # 检查输出文件是否生成
                    self.assertTrue(os.path.exists(output_file), "Output file not generated")
                    
                    # 检查Excel文件是否包含预期的sheet
                    with pd.ExcelFile(output_file) as xls:
                        self.assertIn("predicted_drugs", xls.sheet_names)
                        
                except Exception as e:
                    self.fail(f"predict_drugs failed with error: {str(e)}")
            else:
                self.skipTest("No disease entities found in data")
    
    def test_annotate_shared_genes_pathways(self):
        """测试共享基因通路注释功能"""
        if not all(os.path.exists(f) for f in [self.entity_file, self.knowledge_graph]):
            self.skipTest("Data files not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建一个模拟的预测药物文件
            pred_file = os.path.join(temp_dir, "test_predicted_drugs.xlsx")
            test_drugs = pd.DataFrame({
                "drug_id": ["MESH:C123456", "MESH:C789012"],
                "drug_name": ["Test Drug 1", "Test Drug 2"],
                "score": [0.8, 0.7]
            })
            test_drugs.to_excel(pred_file, sheet_name="predicted_drugs", index=False)
            
            output_file = os.path.join(temp_dir, "test_shared_genes.xlsx")
            
            try:
                self.core.annotate_shared_genes_pathways(
                    predicted_drug_file=pred_file,
                    disease_id="MESH:D123456",
                    knowledge_graph=self.knowledge_graph,
                    output_file=output_file
                )
                
                self.assertTrue(os.path.exists(output_file), "Output file not generated")
                
            except Exception as e:
                self.fail(f"annotate_shared_genes_pathways failed with error: {str(e)}")
    
    def test_annotate_shared_diseases(self):
        """测试共享疾病注释功能"""
        if not all(os.path.exists(f) for f in [self.entity_file, self.knowledge_graph, 
                                              self.entity_embeddings, self.relation_embeddings]):
            self.skipTest("Data files not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建一个模拟的预测药物文件
            pred_file = os.path.join(temp_dir, "test_predicted_drugs.xlsx")
            test_drugs = pd.DataFrame({
                "drug_id": ["MESH:C123456", "MESH:C789012"],
                "drug_name": ["Test Drug 1", "Test Drug 2"],
                "score": [0.8, 0.7]
            })
            test_drugs.to_excel(pred_file, sheet_name="predicted_drugs", index=False)
            
            output_file = os.path.join(temp_dir, "test_shared_diseases.xlsx")
            
            try:
                self.core.annotate_shared_diseases(
                    predicted_drug_file=pred_file,
                    disease_id="MESH:D123456",
                    knowledge_graph=self.knowledge_graph,
                    entity_embeddings=self.entity_embeddings,
                    relation_embeddings=self.relation_embeddings,
                    output_file=output_file,
                    model='TransE_l2',
                    gamma=12.0,
                    top_n=10
                )
                
                self.assertTrue(os.path.exists(output_file), "Output file not generated")
                
            except Exception as e:
                self.fail(f"annotate_shared_diseases failed with error: {str(e)}")
    
    def test_annotate_network_features(self):
        """测试网络特征注释功能"""
        if not all(os.path.exists(f) for f in [self.entity_file, self.knowledge_graph]):
            self.skipTest("Data files not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建一个模拟的预测药物文件
            pred_file = os.path.join(temp_dir, "test_predicted_drugs.xlsx")
            test_drugs = pd.DataFrame({
                "drug_id": ["MESH:C123456", "MESH:C789012"],
                "drug_name": ["Test Drug 1", "Test Drug 2"],
                "score": [0.8, 0.7]
            })
            test_drugs.to_excel(pred_file, sheet_name="predicted_drugs", index=False)
            
            output_file = os.path.join(temp_dir, "test_network_features.xlsx")
            
            try:
                self.core.annotate_network_features(
                    predicted_drug_file=pred_file,
                    disease_id="MESH:D123456",
                    knowledge_graph=self.knowledge_graph,
                    entity_file=self.entity_file,
                    output_file=output_file
                )
                
                self.assertTrue(os.path.exists(output_file), "Output file not generated")
                
            except Exception as e:
                self.fail(f"annotate_network_features failed with error: {str(e)}")
    
    def test_merge_annotations(self):
        """测试注释合并功能"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建模拟的输入文件
            pred_file = os.path.join(temp_dir, "test_predicted_drugs.xlsx")
            shared_genes_file = os.path.join(temp_dir, "test_shared_genes.xlsx")
            shared_diseases_file = os.path.join(temp_dir, "test_shared_diseases.xlsx")
            network_file = os.path.join(temp_dir, "test_network_features.xlsx")
            
            # 创建测试数据
            test_drugs = pd.DataFrame({
                "drug_id": ["MESH:C123456", "MESH:C789012"],
                "drug_name": ["Test Drug 1", "Test Drug 2"],
                "score": [0.8, 0.7]
            })
            
            for file_path in [pred_file, shared_genes_file, shared_diseases_file, network_file]:
                test_drugs.to_excel(file_path, sheet_name="test_sheet", index=False)
            
            output_file = os.path.join(temp_dir, "test_merged_annotations.xlsx")
            
            try:
                self.core.merge_annotations(
                    pred_xlsx=pred_file,
                    shared_genes_xlsx=shared_genes_file,
                    shared_diseases_xlsx=shared_diseases_file,
                    network_anno_xlsx=network_file,
                    output_file=output_file
                )
                
                self.assertTrue(os.path.exists(output_file), "Output file not generated")
                
            except Exception as e:
                self.fail(f"merge_annotations failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main() 