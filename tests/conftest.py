"""
Pytest configuration file for the drugs4disease package tests
"""

import pytest
import tempfile
import os
import shutil
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for the test session"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mock_data_files(temp_dir):
    """Create mock data files for testing"""
    # Mock entity file
    entity_data = {
        'id': ['MESH:D001', 'MESH:D002', 'MESH:C001', 'MESH:C002', 'ENTREZ:123', 'ENTREZ:456'],
        'name': ['Disease1', 'Disease2', 'Drug1', 'Drug2', 'Gene1', 'Gene2'],
        'label': ['Disease', 'Disease', 'Compound', 'Compound', 'Gene', 'Gene']
    }
    entity_file = os.path.join(temp_dir, 'annotated_entities.tsv')
    pd.DataFrame(entity_data).to_csv(entity_file, sep='\t', index=False)
    
    # Mock knowledge graph
    kg_data = {
        'source_id': ['MESH:C001', 'MESH:C002', 'ENTREZ:123', 'ENTREZ:456'],
        'source_type': ['Compound', 'Compound', 'Gene', 'Gene'],
        'source_name': ['Drug1', 'Drug2', 'Gene1', 'Gene2'],
        'target_id': ['MESH:D001', 'MESH:D002', 'MESH:C001', 'MESH:C002'],
        'target_type': ['Disease', 'Disease', 'Compound', 'Compound'],
        'target_name': ['Disease1', 'Disease2', 'Drug1', 'Drug2'],
        'relation_type': ['GNBR::T::Compound:Disease', 'GNBR::T::Compound:Disease', 'GNBR::I::Gene:Compound', 'GNBR::I::Gene:Compound']
    }
    knowledge_graph = os.path.join(temp_dir, 'knowledge_graph.tsv')
    pd.DataFrame(kg_data).to_csv(knowledge_graph, sep='\t', index=False)
    
    # Mock entity embeddings
    embedding_data = {
        'entity_id': ['MESH:D001', 'MESH:D002', 'MESH:C001', 'MESH:C002'],
        'entity_type': ['Disease', 'Disease', 'Compound', 'Compound'],
        'embedding': ['0.1|0.2|0.3', '0.4|0.5|0.6', '0.7|0.8|0.9', '1.0|1.1|1.2']
    }
    entity_embeddings = os.path.join(temp_dir, 'entity_embeddings.tsv')
    pd.DataFrame(embedding_data).to_csv(entity_embeddings, sep='\t', index=False)
    
    # Mock relation embeddings
    relation_data = {
        'relation_type': ['GNBR::T::Compound:Disease', 'GNBR::I::Gene:Compound'],
        'embedding': ['0.1|0.2|0.3', '0.4|0.5|0.6']
    }
    relation_embeddings = os.path.join(temp_dir, 'relation_type_embeddings.tsv')
    pd.DataFrame(relation_data).to_csv(relation_embeddings, sep='\t', index=False)
    
    return {
        'entity_file': entity_file,
        'knowledge_graph': knowledge_graph,
        'entity_embeddings': entity_embeddings,
        'relation_embeddings': relation_embeddings
    }


@pytest.fixture(scope="session")
def mock_test_data():
    """Create mock test data for visualization and filtering tests"""
    test_df = pd.DataFrame({
        'drug_id': ['MESH:C001', 'MESH:C002', 'MESH:C003', 'MESH:C004', 'MESH:C005'],
        'drug_name': ['Drug1', 'Drug2', 'Drug3', 'Drug4', 'Drug5'],
        'score': [0.8, 0.6, 0.9, 0.4, 0.7],
        'pvalue': [0.01, 0.05, 0.001, 0.1, 0.02],
        'num_of_shared_genes_in_path': [5, 2, 8, 1, 3],
        'num_of_shared_pathways': [3, 1, 4, 0, 2],
        'num_of_shared_diseases': [2, 1, 3, 0, 1],
        'drug_degree': [15, 8, 20, 5, 12],
        'drug_betweenness': [0.1, 0.05, 0.15, 0.02, 0.08],
        'drug_closeness': [0.8, 0.6, 0.9, 0.4, 0.7],
        'drug_eigenvector': [0.9, 0.4, 0.95, 0.3, 0.6],
        'num_of_key_genes': [2, 1, 3, 0, 1],
        'existing': [False, True, False, True, False],
        'shared_gene_names': ['Gene1,Gene2', 'Gene1', 'Gene1,Gene2,Gene3', '', 'Gene1'],
        'shared_pathway_names': ['Pathway1,Pathway2', 'Pathway1', 'Pathway1,Pathway2,Pathway3', '', 'Pathway1'],
        'shared_disease_names': ['Disease1,Disease2', 'Disease1', 'Disease1,Disease2,Disease3', '', 'Disease1'],
        'key_gene_names': ['Gene1,Gene2', 'Gene1', 'Gene1,Gene2,Gene3', '', 'Gene1']
    })
    
    filtered_df = test_df[test_df['score'] >= 0.6].copy()
    
    return {
        'full_df': test_df,
        'filtered_df': filtered_df
    }


@pytest.fixture(scope="function")
def temp_file():
    """Create a temporary file for individual tests"""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_file = f.name
    
    yield temp_file
    
    # Clean up
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture(scope="function")
def temp_excel_file():
    """Create a temporary Excel file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        temp_file = f.name
    
    yield temp_file
    
    # Clean up
    if os.path.exists(temp_file):
        os.unlink(temp_file)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Mark tests based on their names
        if "test_full_pipeline" in item.name:
            item.add_marker(pytest.mark.integration)
        elif "test_core" in item.name and "test_run_full_pipeline" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit) 