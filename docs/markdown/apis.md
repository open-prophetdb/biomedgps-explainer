# BioMedGPS Explainer API Documentation

## Overview

This document provides comprehensive API documentation for the BioMedGPS Explainer toolkit, covering all major classes, methods, and their usage.

## Core Classes

### DrugDiseaseCore

The main class responsible for drug discovery analysis using knowledge graph embeddings.

#### Constructor
```python
DrugDiseaseCore()
```

#### Methods

##### run_full_pipeline()
```python
run_full_pipeline(
    disease_id: str,
    entity_file: Optional[str] = None,
    knowledge_graph: Optional[str] = None,
    entity_embeddings: Optional[str] = None,
    relation_embeddings: Optional[str] = None,
    output_dir: Optional[str] = None,
    model: str = 'TransE_l2',
    top_n_diseases: int = 100,
    gamma: float = 12.0,
    threshold: float = 0.5,
    relation_type: str = 'GNBR::T::Compound:Disease',
    top_n_drugs: int = 1000
) -> None
```

**Description**: One-click complete analysis pipeline that generates annotated_drugs.xlsx with all analysis results.

**Parameters**:
- `disease_id`: Disease identifier (e.g., "MONDO:0004979")
- `entity_file`: Path to entity annotations file (optional, uses default if not specified)
- `knowledge_graph`: Path to knowledge graph file (optional, uses default if not specified)
- `entity_embeddings`: Path to entity embeddings file (optional, uses default if not specified)
- `relation_embeddings`: Path to relation embeddings file (optional, uses default if not specified)
- `output_dir`: Output directory for results
- `model`: KGE model type (default: 'TransE_l2')
- `top_n_diseases`: Number of similar diseases to consider (default: 100)
- `gamma`: Margin parameter for KGE training (default: 12.0)
- `threshold`: Drug filtering threshold (default: 0.5)
- `relation_type`: Relation type for drug-disease associations (default: 'GNBR::T::Compound:Disease')
- `top_n_drugs`: Number of drugs to analyze (default: 1000)

**Returns**: None

**Example**:
```python
core = DrugDiseaseCore()
core.run_full_pipeline(
    disease_id="MONDO:0004979",
    output_dir="results/",
    model='TransE_l2',
    top_n_diseases=50,
    gamma=12.0,
    threshold=0.5,
    top_n_drugs=100
)
```

##### predict_drugs()
```python
predict_drugs(
    disease_id: str,
    entity_file: str,
    knowledge_graph: str,
    entity_embeddings: str,
    relation_embeddings: str,
    model: str,
    top_n_diseases: int,
    gamma: float,
    threshold: float,
    relation_type: str,
    output_file: str
) -> None
```

**Description**: Generate potential drug list using KGE models.

**Parameters**:
- `disease_id`: Target disease identifier
- `entity_file`: Path to entity annotations file
- `knowledge_graph`: Path to knowledge graph file
- `entity_embeddings`: Path to entity embeddings file
- `relation_embeddings`: Path to relation embeddings file
- `model`: KGE model type
- `top_n_diseases`: Number of similar diseases
- `gamma`: Margin parameter
- `threshold`: Prediction threshold
- `relation_type`: Relation type for drug-disease associations
- `output_file`: Output Excel file path

##### get_disease_name()
```python
get_disease_name(disease_id: str, entity_file: str) -> str
```

**Description**: Get disease name from disease ID.

**Parameters**:
- `disease_id`: Disease identifier
- `entity_file`: Path to entity annotations file

**Returns**: Disease name as string

##### get_drug_names()
```python
get_drug_names(drug_ids: List[str], entity_file: str) -> List[str]
```

**Description**: Get drug names from drug IDs.

**Parameters**:
- `drug_ids`: List of drug identifiers
- `entity_file`: Path to entity annotations file

**Returns**: List of drug names

### DrugFilter

Class for filtering drug candidates based on various criteria.

#### Constructor
```python
DrugFilter()
```

#### Methods

##### filter_drugs()
```python
filter_drugs(
    input_file: str,
    expression: str,
    output_file: str,
    sheet_names: Tuple[str, str] = ("annotated_drugs", "filtered_drugs")
) -> None
```

**Description**: Filter drugs based on logical expressions.

**Parameters**:
- `input_file`: Input Excel file path
- `expression`: Filter expression (e.g., "score > 0.6 and existing == False")
- `output_file`: Output Excel file path
- `sheet_names`: Tuple of (input_sheet, output_sheet) names

**Example**:
```python
filter = DrugFilter()
filter.filter_drugs(
    input_file="results/annotated_drugs.xlsx",
    expression="score > 0.7 and num_of_shared_genes_in_path >= 1",
    output_file="results/filtered_drugs.xlsx"
)
```

**Supported Filter Expressions**:
- Numerical comparisons: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Logical operators: `and`, `or`, `not`
- Boolean fields: `existing`, `is_key_gene`
- String matching and pattern matching

### Visualizer

Class for generating comprehensive visualizations and reports.

#### Constructor
```python
Visualizer(disease_id: str, disease_name: str, embed_images: bool = True)
```

**Parameters**:
- `disease_id`: Disease identifier
- `disease_name`: Disease name
- `embed_images`: Whether to embed images in HTML report (default: True)

#### Methods

##### create_visualization()
```python
create_visualization(
    data_file: str,
    viz_type: str,
    output_file: str,
    sheet_names: Tuple[str, str] = ("annotated_drugs", "filtered_drugs")
) -> str
```

**Description**: Generate specific visualization chart.

**Parameters**:
- `data_file`: Input data file path
- `viz_type`: Visualization type (see supported types below)
- `output_file`: Output file path
- `sheet_names`: Tuple of sheet names for input data

**Returns**: Interpretation text for the visualization

**Supported Visualization Types**:
- `score_distribution`: Predicted score distribution
- `predicted_score_boxplot`: Score distribution by knowledge graph inclusion
- `disease_similarity_heatmap`: Drug similarity heatmap
- `network_centrality`: Network centrality analysis
- `shared_genes_pathways`: Shared genes and pathways analysis
- `drug_disease_network`: Drug similarity network
- `shared_gene_count`: Shared gene count distribution
- `score_vs_degree`: Score vs degree relationship
- `shared_gene_count_vs_score`: Shared gene count vs score scatter plot
- `overlap_pathways`: Overlapping pathways distribution
- `key_genes_distribution`: Key genes distribution
- `existing_vs_predicted`: Known vs predicted drugs ratio
- `prompt`: Prompt generation for AI research

##### generate_report()
```python
generate_report(
    data_file: str,
    output_file: str,
    title: str = "Drug Discovery Analysis Report",
    sheet_names: Tuple[str, str] = ("annotated_drugs", "filtered_drugs")
) -> str
```

**Description**: Generate comprehensive HTML report with all visualizations.

**Parameters**:
- `data_file`: Input data file path
- `output_file`: Output HTML file path
- `title`: Report title
- `sheet_names`: Tuple of sheet names for input data

**Returns**: Status message

**Example**:
```python
visualizer = Visualizer()
visualizer.generate_report(
    data_file="results/filtered_drugs.xlsx",
    output_file="results/analysis_report.html",
    title="Drug Discovery Analysis Report - Asthma"
)
```

##### get_chart_types()
```python
get_chart_types() -> List[str]
```

**Description**: Get list of all supported chart types.

**Returns**: List of supported visualization types

##### get_chart_title()
```python
get_chart_title(chart_type: str) -> str
```

**Description**: Get human-readable title for chart type.

**Parameters**:
- `chart_type`: Chart type identifier

**Returns**: Human-readable chart title

### DrugExplain

Class for generating AI-powered drug explanations.

#### Constructor
```python
DrugExplain()
```

#### Methods

##### generate_prompt()
```python
generate_prompt(
    drug_names: List[str],
    disease_name: str
) -> str
```

**Description**: Generate AI prompt for drug explanation.

**Parameters**:
- `drug_names`: List of drug names to explain
- `disease_name`: Target disease name

**Returns**: Formatted prompt string for AI model

## Utility Functions

### get_model_file_paths()
```python
get_model_file_paths(
    entity_file: Optional[str] = None,
    knowledge_graph: Optional[str] = None,
    entity_embeddings: Optional[str] = None,
    relation_embeddings: Optional[str] = None,
    project_root: Optional[str] = None
) -> Tuple[str, str, str, str]
```

**Description**: Smartly get model file paths with automatic ZIP decompression.

**Parameters**:
- `entity_file`: Entity file path (optional)
- `knowledge_graph`: Knowledge graph file path (optional)
- `entity_embeddings`: Entity embeddings file path (optional)
- `relation_embeddings`: Relation embeddings file path (optional)
- `project_root`: Project root directory (optional)

**Returns**: Tuple of four file paths (entity_file, knowledge_graph, entity_embeddings, relation_embeddings)

### validate_model_files()
```python
validate_model_files(
    entity_file: str,
    knowledge_graph: str,
    entity_embeddings: str,
    relation_embeddings: str
) -> bool
```

**Description**: Validate that all required model files exist and are accessible.

**Parameters**:
- `entity_file`: Entity file path
- `knowledge_graph`: Knowledge graph file path
- `entity_embeddings`: Entity embeddings file path
- `relation_embeddings`: Relation embeddings file path

**Returns**: True if all files are valid, False otherwise

### extract_model_files_if_needed()
```python
extract_model_files_if_needed(model_dir: str) -> bool
```

**Description**: Automatically extract ZIP files if needed.

**Parameters**:
- `model_dir`: Model directory path

**Returns**: True if extraction successful or not needed, False otherwise

## Command Line Interface

### Main Commands

#### run
```bash
biomedgps-explainer run --disease-id MONDO:0004979 --output-dir results/
```

**Options**:
- `--disease-id`: Disease ID (required)
- `--model-run-id`: Model run ID (default: 6vlvgvfq)
- `--output-dir`: Output directory (required)
- `--top-n-diseases`: Number of similar diseases (default: 100)
- `--threshold`: Drug filtering threshold (default: 0.5)
- `--relation-type`: Relation type (default: GNBR::T::Compound:Disease)
- `--top-n-drugs`: Number of drugs to analyze (default: 1000)

#### filter
```bash
biomedgps-explainer filter --input-file results/annotated_drugs.xlsx --expression "score > 0.6" --output-file results/filtered_drugs.xlsx
```

**Options**:
- `--input-file`: Input Excel file (required)
- `--expression`: Filter expression (required)
- `--output-file`: Output Excel file (required)

#### visualize
```bash
biomedgps-explainer visualize --input-file results/annotated_drugs.xlsx --output-dir results/visualizations/ --disease-id MONDO:0004979 --disease-name "asthma"
```

**Options**:
- `--input-file`: Input Excel file (required)
- `--output-dir`: Output directory (required)
- `--viz-type`: Visualization type (default: all)
- `--disease-id`: Disease ID (required)
- `--disease-name`: Disease name (required)

#### pipeline
```bash
biomedgps-explainer pipeline --disease-id MONDO:0004979 --output-dir results/ --filter-expression "score > 0.6 and existing == False"
```

**Description**: Execute complete workflow (run → filter → visualize) in a single command.

**Options**:
- `--disease-id`: Disease ID (required)
- `--model-run-id`: Model run ID (default: 6vlvgvfq)
- `--filter-expression`: Filter expression (optional)
- `--output-dir`: Output directory (default: results)
- `--top-n-diseases`: Number of similar diseases (default: 100)
- `--threshold`: Drug filtering threshold (default: 0.5)
- `--relation-type`: Relation type (default: GNBR::T::Compound:Disease)
- `--top-n-drugs`: Number of drugs to interpret (default: 100)

## Data Structures

### Input Data Format

#### Entity File (annotated_entities.tsv)
```tsv
id  label   name
MONDO:0004979  Disease  asthma
CHEBI:12345    Compound aspirin
```

#### Knowledge Graph File (knowledge_graph.tsv)
```tsv
source_id  source_type  source_name  target_id  target_type  target_name  relation_type
CHEBI:12345  Compound  aspirin  MONDO:0004979  Disease  asthma  GNBR::T::Compound:Disease
```

#### Entity Embeddings File (entity_embeddings.tsv)
```tsv
entity_id  entity_type  embedding
MONDO:0004979  Disease  0.1|0.2|0.3|...
CHEBI:12345    Compound 0.4|0.5|0.6|...
```

#### Relation Embeddings File (relation_type_embeddings.tsv)
```tsv
relation_id  embedding
GNBR::T::Compound:Disease  0.1|0.2|0.3|...
```

### Output Data Format

#### Annotated Drugs Excel File
Contains the following columns:
- `drug_id`: Drug identifier
- `drug_name`: Drug name
- `score`: Predicted score
- `existing`: Whether drug is known for the disease
- `num_of_shared_genes_in_path`: Number of shared genes
- `num_of_shared_pathways`: Number of overlapping pathways
- `drug_degree`: Network degree
- `num_of_key_genes`: Number of key genes
- `pvalue`: Statistical significance
- `shared_gene_names`: Names of shared genes
- `shared_pathways`: Names of overlapping pathways
- `shared_disease_names`: Names of shared diseases

## Error Handling

### Common Exceptions

#### FileNotFoundError
Raised when required model files are missing.

#### ValueError
Raised for invalid parameters or data formats.

#### RuntimeError
Raised for analysis failures or insufficient data.

### Error Recovery

The toolkit includes automatic error recovery for:
- Missing ZIP files (automatic decompression)
- Invalid file paths (automatic path resolution)
- Memory issues (automatic parameter adjustment)

## Performance Considerations

### Memory Usage
- Large datasets may require significant memory
- Use smaller `top_n_drugs` values for memory-constrained environments
- Consider processing in batches for very large datasets

### Processing Time
- Drug prediction: O(n_drugs × n_diseases)
- Network analysis: O(n_drugs²)
- Pathway enrichment: O(n_drugs × n_pathways)

### Optimization Tips
- Use appropriate `top_n_diseases` values
- Filter early in the pipeline
- Use parallel processing for independent analyses 