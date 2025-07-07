import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from drugs4disease.core import DrugDiseaseCore


class TestDrugDiseaseCore(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.core = DrugDiseaseCore()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock entity file with xrefs column
        self.entity_data = {
            'id': ['MESH:D001', 'MESH:D002', 'MESH:D003', 'MESH:C001', 'MESH:C002', 'MESH:C003', 'ENTREZ:123', 'ENTREZ:456'],
            'label': ['Disease', 'Disease', 'Disease', 'Compound', 'Compound', 'Compound', 'Gene', 'Gene'],
            'name': ['Disease1', 'Disease2', 'Disease3', 'Drug1', 'Drug2', 'Drug3', 'Gene1', 'Gene2'],
            'xrefs': [
                'MESH:D001|UMLS:C001',
                'MESH:D002|UMLS:C002',
                'MESH:D003|UMLS:C003',
                'MESH:C001|UMLS:C004',
                'MESH:C002|UMLS:C005',
                'MESH:C003|UMLS:C006',
                'ENTREZ:123|SYMBOL:GENE1',
                'ENTREZ:456|SYMBOL:GENE2'
            ]
        }
        
        # Create mock knowledge graph data
        self.kg_data = {
            'source_id': ['MESH:D001', 'MESH:D002', 'MESH:D003', 'MESH:C001', 'MESH:C002', 'MESH:C003', 'ENTREZ:123'],
            'source_type': ['Disease', 'Disease', 'Disease', 'Compound', 'Compound', 'Compound', 'Gene'],
            'source_name': ['Disease1', 'Disease2', 'Disease3', 'Drug1', 'Drug2', 'Drug3', 'Gene1'],
            'target_id': ['ENTREZ:123', 'ENTREZ:456', 'ENTREZ:123', 'MESH:D001', 'MESH:D002', 'MESH:D003', 'MESH:C001'],
            'target_type': ['Gene', 'Gene', 'Gene', 'Disease', 'Disease', 'Disease', 'Compound'],
            'target_name': ['Gene1', 'Gene2', 'Gene1', 'Disease1', 'Disease2', 'Disease3', 'Drug1'],
            'relation_type': [
                'BioMedGPS::AssociatedWith::Disease:Gene',
                'BioMedGPS::AssociatedWith::Disease:Gene',
                'BioMedGPS::AssociatedWith::Disease:Gene',
                'GNBR::T::Compound:Disease',
                'GNBR::T::Compound:Disease',
                'GNBR::T::Compound:Disease',
                'BioMedGPS::Targets::Gene:Compound'
            ]
        }
        
        # Create mock entity embeddings data
        self.entity_embeddings_data = {
            'entity_id': ['MESH:D001', 'MESH:D002', 'MESH:D003', 'MESH:C001', 'MESH:C002', 'MESH:C003', 'ENTREZ:123'],
            'entity_type': ['Disease', 'Disease', 'Disease', 'Compound', 'Compound', 'Compound', 'Gene'],
            'entity_name': ['Disease1', 'Disease2', 'Disease3', 'Drug1', 'Drug2', 'Drug3', 'Gene1'],
            'embedding': [
                '0.1|0.2|0.3',
                '0.2|0.3|0.4',
                '0.3|0.4|0.5',
                '0.4|0.5|0.6',
                '0.7|0.8|0.9',
                '0.1|0.2|0.3',
                '0.4|0.5|0.6'
            ]
        }
        
        # Create mock relation embeddings data
        self.relation_embeddings_data = {
            'id': ['BioMedGPS::AssociatedWith::Disease:Gene', 'GNBR::T::Compound:Disease'],
            'embedding': ['0.1|0.2|0.3', '0.4|0.5|0.6']
        }
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test DrugDiseaseCore initialization"""
        core = DrugDiseaseCore()
        self.assertIsNotNone(core.logger)
        self.assertEqual(core.logger.name, "DrugDiseaseCore")
    
    def test_get_disease_name(self):
        """Test getting disease name from entity file"""
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        pd.DataFrame(self.entity_data).to_csv(entity_file, sep='\t', index=False)
        
        disease_name = self.core.get_disease_name('MESH:D001', entity_file)
        self.assertEqual(disease_name, 'Disease1')
    
    def test_get_disease_name_not_found(self):
        """Test getting disease name when disease ID doesn't exist"""
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        pd.DataFrame(self.entity_data).to_csv(entity_file, sep='\t', index=False)
        
        with self.assertRaises(IndexError):
            self.core.get_disease_name('MESH:D999', entity_file)
    
    def test_get_drug_names(self):
        """Test getting drug names from entity file"""
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        pd.DataFrame(self.entity_data).to_csv(entity_file, sep='\t', index=False)
        
        drug_names = self.core.get_drug_names(['MESH:C001'], entity_file)
        self.assertEqual(drug_names, ['Drug1'])
    
    def test_get_drug_names_empty_list(self):
        """Test getting drug names with empty list"""
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        pd.DataFrame(self.entity_data).to_csv(entity_file, sep='\t', index=False)
        
        drug_names = self.core.get_drug_names([], entity_file)
        self.assertEqual(drug_names, [])
    
    @patch('drugs4disease.core.lib.kge_score_fn_batch')
    def test_predict_drugs(self, mock_kge_score):
        """Test drug prediction functionality"""
        # Mock the KGE score function to return scores for each drug
        mock_kge_score.return_value = np.array([0.8, 0.6, 0.4])  # 3 drugs in mock data
        
        # Create required files
        pd.DataFrame(self.kg_data).to_csv(os.path.join(self.temp_dir, 'kg.tsv'), sep='\t', index=False)
        pd.DataFrame(self.entity_data).to_csv(os.path.join(self.temp_dir, 'entity.tsv'), sep='\t', index=False)
        pd.DataFrame(self.entity_embeddings_data).to_csv(os.path.join(self.temp_dir, 'entity_emb.tsv'), sep='\t', index=False)
        pd.DataFrame(self.relation_embeddings_data).to_csv(os.path.join(self.temp_dir, 'relation_emb.tsv'), sep='\t', index=False)
        
        output_file = os.path.join(self.temp_dir, 'predicted_drugs.xlsx')
        
        self.core.predict_drugs(
            disease_id='MESH:D001',
            entity_file=os.path.join(self.temp_dir, 'entity.tsv'),
            knowledge_graph=os.path.join(self.temp_dir, 'kg.tsv'),
            entity_embeddings=os.path.join(self.temp_dir, 'entity_emb.tsv'),
            relation_embeddings=os.path.join(self.temp_dir, 'relation_emb.tsv'),
            model='TransE_l2',
            top_n_diseases=10,
            gamma=12.0,
            threshold=0.5,
            relation_type='GNBR::T::Compound:Disease',
            output_file=output_file
        )
        
        # Check if output file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check if the file contains expected data
        df = pd.read_excel(output_file, sheet_name='predicted_drugs')
        self.assertIn('drug_id', df.columns)
        self.assertIn('drug_name', df.columns)
        self.assertIn('score', df.columns)
        self.assertIn('rank', df.columns)
        self.assertIn('pvalue', df.columns)
    
    def test_predict_drugs_invalid_output_format(self):
        """Test drug prediction with invalid output format"""
        output_file = os.path.join(self.temp_dir, 'predicted_drugs.txt')
        
        with self.assertRaises(ValueError):
            self.core.predict_drugs(
                disease_id='MESH:D001',
                entity_file=os.path.join(self.temp_dir, 'entity.tsv'),
                knowledge_graph=os.path.join(self.temp_dir, 'kg.tsv'),
                entity_embeddings=os.path.join(self.temp_dir, 'entity_emb.tsv'),
                relation_embeddings=os.path.join(self.temp_dir, 'relation_emb.tsv'),
                model='TransE_l2',
                top_n_diseases=10,
                gamma=12.0,
                threshold=0.5,
                relation_type='GNBR::T::Compound:Disease',
                output_file=output_file
            )
    
    def test_predict_drugs_invalid_knowledge_graph_format(self):
        """Test drug prediction with invalid knowledge graph format"""
        # Create invalid knowledge graph
        invalid_kg_data = {
            'source_id': ['MESH:C001'],
            'target_id': ['MESH:D001']
            # Missing required columns
        }
        invalid_kg_file = os.path.join(self.temp_dir, 'invalid_kg.tsv')
        pd.DataFrame(invalid_kg_data).to_csv(invalid_kg_file, sep='\t', index=False)
        
        output_file = os.path.join(self.temp_dir, 'predicted_drugs.xlsx')
        
        with self.assertRaises(ValueError):
            self.core.predict_drugs(
                disease_id='MESH:D001',
                entity_file=os.path.join(self.temp_dir, 'entity.tsv'),
                knowledge_graph=invalid_kg_file,
                entity_embeddings=os.path.join(self.temp_dir, 'entity_emb.tsv'),
                relation_embeddings=os.path.join(self.temp_dir, 'relation_emb.tsv'),
                model='TransE_l2',
                top_n_diseases=10,
                gamma=12.0,
                threshold=0.5,
                relation_type='GNBR::T::Compound:Disease',
                output_file=output_file
            )
    
    def test_predict_drugs_disease_not_found(self):
        """Test drug prediction with disease ID not in embeddings"""
        # Create required files
        pd.DataFrame(self.kg_data).to_csv(os.path.join(self.temp_dir, 'kg.tsv'), sep='\t', index=False)
        pd.DataFrame(self.entity_data).to_csv(os.path.join(self.temp_dir, 'entity.tsv'), sep='\t', index=False)
        pd.DataFrame(self.entity_embeddings_data).to_csv(os.path.join(self.temp_dir, 'entity_emb.tsv'), sep='\t', index=False)
        pd.DataFrame(self.relation_embeddings_data).to_csv(os.path.join(self.temp_dir, 'relation_emb.tsv'), sep='\t', index=False)
        
        output_file = os.path.join(self.temp_dir, 'predicted_drugs.xlsx')
        
        with self.assertRaises(ValueError):
            self.core.predict_drugs(
                disease_id='MESH:D999',  # Non-existent disease
                entity_file=os.path.join(self.temp_dir, 'entity.tsv'),
                knowledge_graph=os.path.join(self.temp_dir, 'kg.tsv'),
                entity_embeddings=os.path.join(self.temp_dir, 'entity_emb.tsv'),
                relation_embeddings=os.path.join(self.temp_dir, 'relation_emb.tsv'),
                model='TransE_l2',
                top_n_diseases=10,
                gamma=12.0,
                threshold=0.5,
                relation_type='GNBR::T::Compound:Disease',
                output_file=output_file
            )
    
    def test_annotate_shared_genes_pathways(self):
        """Test shared genes and pathways annotation"""
        # Create mock files
        kg_file = os.path.join(self.temp_dir, 'kg.tsv')
        pred_file = os.path.join(self.temp_dir, 'predicted_drugs.xlsx')
        output_file = os.path.join(self.temp_dir, 'shared_genes_pathways.xlsx')
        
        pd.DataFrame(self.kg_data).to_csv(kg_file, sep='\t', index=False)
        
        # Create predicted drugs file
        pred_data = {
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2']
        }
        with pd.ExcelWriter(pred_file) as writer:
            pd.DataFrame(pred_data).to_excel(writer, sheet_name='predicted_drugs', index=False)
        
        # Mock the annotation logic
        with patch('drugs4disease.core.pd.read_csv') as mock_read_csv, \
             patch('drugs4disease.core.pd.read_excel') as mock_read_excel, \
             patch('drugs4disease.core.lib.save_df') as mock_save:
            
            # Mock the knowledge graph data
            mock_kg_df = pd.DataFrame(self.kg_data)
            mock_read_csv.return_value = mock_kg_df
            
            # Mock the predicted drugs data
            mock_pred_df = pd.DataFrame(pred_data)
            mock_read_excel.return_value = mock_pred_df
            
            self.core.annotate_shared_genes_pathways(
                predicted_drug_file=pred_file,
                disease_id='MESH:D001',
                knowledge_graph=kg_file,
                output_file=output_file,
                top_n_drugs=100
            )
            
            # Verify save was called
            mock_save.assert_called()
    
    def test_annotate_shared_diseases(self):
        """Test shared diseases annotation"""
        # Create mock files
        kg_file = os.path.join(self.temp_dir, 'kg.tsv')
        entity_emb_file = os.path.join(self.temp_dir, 'entity_emb.tsv')
        relation_emb_file = os.path.join(self.temp_dir, 'relation_emb.tsv')
        pred_file = os.path.join(self.temp_dir, 'predicted_drugs.xlsx')
        output_file = os.path.join(self.temp_dir, 'shared_diseases.xlsx')
        
        pd.DataFrame(self.kg_data).to_csv(kg_file, sep='\t', index=False)
        pd.DataFrame(self.entity_embeddings_data).to_csv(entity_emb_file, sep='\t', index=False)
        
        # Create relation embeddings with required relation types
        relation_data = {
            'id': ['BioMedGPS::SimilarWith::Disease:Disease', 'Hetionet::DrD::Disease:Disease'],
            'embedding': ['0.1|0.2|0.3', '0.4|0.5|0.6']
        }
        pd.DataFrame(relation_data).to_csv(relation_emb_file, sep='\t', index=False)
        
        # Create predicted drugs file
        pred_data = {
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2']
        }
        with pd.ExcelWriter(pred_file) as writer:
            pd.DataFrame(pred_data).to_excel(writer, sheet_name='predicted_drugs', index=False)
        
        with patch('drugs4disease.core.lib.save_df') as mock_save:
            self.core.annotate_shared_diseases(
                predicted_drug_file=pred_file,
                disease_id='MESH:D001',
                knowledge_graph=kg_file,
                entity_embeddings=entity_emb_file,
                relation_embeddings=relation_emb_file,
                output_file=output_file,
                model='TransE_l2',
                gamma=12.0,
                top_n=10
            )
            mock_save.assert_called()
    
    def test_annotate_network_features(self):
        """Test network features annotation"""
        # Create mock files
        kg_file = os.path.join(self.temp_dir, 'kg.tsv')
        entity_file = os.path.join(self.temp_dir, 'entity.tsv')
        pred_file = os.path.join(self.temp_dir, 'predicted_drugs.xlsx')
        output_file = os.path.join(self.temp_dir, 'network_annotations.xlsx')
        
        pd.DataFrame(self.kg_data).to_csv(kg_file, sep='\t', index=False)
        pd.DataFrame(self.entity_data).to_csv(entity_file, sep='\t', index=False)
        
        # Create predicted drugs file
        pred_data = {
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2'],
            'score': [0.8, 0.6],
            'pvalue': [0.01, 0.05]
        }
        with pd.ExcelWriter(pred_file) as writer:
            pd.DataFrame(pred_data).to_excel(writer, sheet_name='predicted_drugs', index=False)
        
        # Mock the annotation logic
        with patch('drugs4disease.core.nx.MultiDiGraph') as mock_graph, \
             patch('drugs4disease.core.lib.save_df') as mock_save:
            
            # Mock the graph with all required methods
            class GraphMock:
                def __init__(self):
                    self.edges = []
                def __contains__(self, item):
                    return True
                def successors(self, node):
                    return iter([])
                def add_edge(self, source, target, relation=None):
                    self.edges.append((source, target, relation))
            mock_graph.return_value = GraphMock()
            
            self.core.annotate_network_features(
                predicted_drug_file=pred_file,
                disease_id='MESH:D001',
                knowledge_graph=kg_file,
                entity_file=entity_file,
                output_file=output_file,
                top_n_drugs=100
            )
            
            # Verify save was called
            mock_save.assert_called()
    
    def test_merge_annotations(self):
        """Test annotation merging"""
        # Create mock files
        pred_file = os.path.join(self.temp_dir, 'predicted_drugs.xlsx')
        shared_genes_file = os.path.join(self.temp_dir, 'shared_genes_pathways.xlsx')
        shared_diseases_file = os.path.join(self.temp_dir, 'shared_diseases.xlsx')
        network_file = os.path.join(self.temp_dir, 'network_annotations.xlsx')
        output_file = os.path.join(self.temp_dir, 'annotated_drugs.xlsx')
        
        # Create predicted drugs data
        pred_data = {
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2'],
            'score': [0.8, 0.6],
            'pvalue': [0.01, 0.05]
        }
        with pd.ExcelWriter(pred_file) as writer:
            pd.DataFrame(pred_data).to_excel(writer, sheet_name='predicted_drugs', index=False)
        
        # Create shared genes data
        shared_genes_data = {
            'predicted_drug_id': ['MESH:C001', 'MESH:C002'],
            'num_of_shared_genes': [2, 1],
            'shared_gene_names': ['Gene1|Gene2', 'Gene1']
        }
        with pd.ExcelWriter(shared_genes_file) as writer:
            pd.DataFrame(shared_genes_data).to_excel(writer, sheet_name='shared_genes', index=False)
        
        # Create shared diseases data
        shared_diseases_data = {
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'num_of_shared_diseases': [1, 0],
            'shared_disease_names': ['Disease1', '']
        }
        with pd.ExcelWriter(shared_diseases_file) as writer:
            pd.DataFrame(shared_diseases_data).to_excel(writer, sheet_name='shared_diseases', index=False)
        
        # Create network annotations data with proper numeric values
        network_data = {
            'drug_id': ['MESH:C001', 'MESH:C002'],
            'drug_name': ['Drug1', 'Drug2'],
            'drug_degree': [10, 5],
            'num_of_key_genes': [2, 1],
            'key_genes': ['Gene1,Gene2', 'Gene1'],
            'num_of_shared_genes_in_path': [3, 1],
            'paths': ['path1|path2', 'path1'],
            'existing': [True, False],
            'num_of_shared_pathways': [2, 1],
            'shared_pathways': ['pathway1|pathway2', 'pathway1'],
            'rank': [1, 2]
        }
        with pd.ExcelWriter(network_file) as writer:
            pd.DataFrame(network_data).to_excel(writer, sheet_name='network_annotations', index=False)
        
        self.core.merge_annotations(
            pred_file, shared_genes_file, shared_diseases_file, network_file, output_file
        )
        
        # Check if output file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check if the file contains expected data
        df = pd.read_excel(output_file, sheet_name='annotated_drugs')
        expected_columns = [
            'drug_id', 'drug_name', 'score', 'pvalue',
            'num_of_shared_genes', 'shared_gene_names',
            'num_of_shared_diseases', 'shared_disease_names',
            'drug_degree', 'num_of_key_genes', 'key_genes'
        ]
        
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    @patch('drugs4disease.core.get_model_file_paths')
    @patch('drugs4disease.core.validate_model_files')
    def test_run_full_pipeline(self, mock_validate, mock_get_paths):
        """Test full pipeline execution"""
        # Mock file paths
        mock_get_paths.return_value = (
            os.path.join(self.temp_dir, 'entity.tsv'),
            os.path.join(self.temp_dir, 'kg.tsv'),
            os.path.join(self.temp_dir, 'entity_emb.tsv'),
            os.path.join(self.temp_dir, 'relation_emb.tsv')
        )
        mock_validate.return_value = True
        
        # Create mock files
        pd.DataFrame(self.entity_data).to_csv(os.path.join(self.temp_dir, 'entity.tsv'), sep='\t', index=False)
        pd.DataFrame(self.kg_data).to_csv(os.path.join(self.temp_dir, 'kg.tsv'), sep='\t', index=False)
        pd.DataFrame(self.entity_embeddings_data).to_csv(os.path.join(self.temp_dir, 'entity_emb.tsv'), sep='\t', index=False)
        pd.DataFrame(self.relation_embeddings_data).to_csv(os.path.join(self.temp_dir, 'relation_emb.tsv'), sep='\t', index=False)
        
        # Mock the individual methods
        with patch.object(self.core, 'predict_drugs') as mock_predict, \
             patch.object(self.core, 'annotate_shared_genes_pathways') as mock_genes, \
             patch.object(self.core, 'annotate_shared_diseases') as mock_diseases, \
             patch.object(self.core, 'annotate_network_features') as mock_network, \
             patch.object(self.core, 'merge_annotations') as mock_merge:
            
            self.core.run_full_pipeline(
                disease_id='MESH:D001',
                output_dir=self.temp_dir
            )
            
            # Verify methods were called
            mock_predict.assert_called_once()
            mock_genes.assert_called_once()
            mock_diseases.assert_called_once()
            mock_network.assert_called_once()
            mock_merge.assert_called_once()


if __name__ == '__main__':
    unittest.main() 