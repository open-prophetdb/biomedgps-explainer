# BioMedGPS Explainer User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Data Preparation](#data-preparation)
5. [Basic Usage](#basic-usage)
6. [Advanced Features](#advanced-features)
7. [Visualization Guide](#visualization-guide)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [FAQ](#faq)

## Introduction

BioMedGPS Explainer is a comprehensive toolkit for drug discovery analysis using knowledge graph embeddings. This guide will walk you through the complete process from installation to generating comprehensive analysis reports.

### What You'll Learn

- How to set up and install the toolkit
- How to prepare and validate your data
- How to run drug discovery analysis
- How to filter and visualize results
- How to interpret the outputs
- How to troubleshoot common issues

## Installation

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB recommended for large datasets)
- **Storage**: At least 5GB free space for model files and results
- **Operating System**: Windows, macOS, or Linux

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd biomedgps-explainer
   ```

2. **Create Virtual Environment (Recommended)**
   ```bash
   python -m venv biomedgps_env
   
   # On Windows
   biomedgps_env\Scripts\activate
   
   # On macOS/Linux
   source biomedgps_env/bin/activate
   ```

3. **Install the Package**
   ```bash
   pip install -e .
   ```

4. **Verify Installation**
   ```bash
   biomedgps-explainer --help
   ```

## Getting Started

### Quick Start Example

1. **Prepare Data Directory**
   ```bash
   mkdir -p data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/
   ```

2. **Add Your Model Files**
   - Place your BioMedGPS model files in the data directory
   - Supports both ZIP and uncompressed TSV formats

3. **Validate Data**
   ```bash
   python3 examples/run_data_validation.py
   ```

4. **Run Complete Analysis**
   ```bash
   python3 examples/run_full_example.py
   ```

5. **View Results**
   - Check the `results/` directory for output files
   - Open `results/visualization_report/analysis_report.html` for the interactive report

## Data Preparation

### Required Files

The toolkit requires four main data files:

1. **annotated_entities.tsv** - Entity annotations
2. **knowledge_graph.tsv** - Knowledge graph triples
3. **entity_embeddings.tsv** - Entity embeddings
4. **relation_type_embeddings.tsv** - Relation embeddings

### File Formats

#### Entity File Format
```tsv
id  label   name
MONDO:0004979  Disease  asthma
CHEBI:12345    Compound aspirin
```

#### Knowledge Graph Format
```tsv
source_id  source_type  source_name  target_id  target_type  target_name  relation_type
CHEBI:12345  Compound  aspirin  MONDO:0004979  Disease  asthma  GNBR::T::Compound:Disease
```

#### Embeddings Format
```tsv
entity_id  entity_type  embedding
MONDO:0004979  Disease  0.1|0.2|0.3|0.4|...
```

### Data Validation

Always validate your data before running analysis:

```bash
python3 examples/run_data_validation.py
```

This will:
- Check file existence and format
- Automatically decompress ZIP files if needed
- Verify data integrity
- Provide detailed error messages if issues are found

## Basic Usage

### Command Line Interface

#### 1. Run Complete Analysis
```bash
biomedgps-explainer run --disease-id MONDO:0004979 --output-dir results/
```

**Parameters**:
- `--disease-id`: Disease ID (required)
- `--output-dir`: Output directory (required)
- `--model-run-id`: Model run ID (default: 6vlvgvfq)
- `--top-n-diseases`: Number of similar diseases (default: 100)
- `--threshold`: Drug filtering threshold (default: 0.5)
- `--relation-type`: Relation type (default: GNBR::T::Compound:Disease)
- `--top-n-drugs`: Number of drugs to interpret (default: 1000)

#### 2. Filter Results
```bash
biomedgps-explainer filter \
  --input-file results/annotated_drugs.xlsx \
  --expression "score > 0.6 and existing == False" \
  --output-file results/filtered_drugs.xlsx
```

#### 3. Generate Visualizations
```bash
biomedgps-explainer visualize \
  --input-file results/filtered_drugs.xlsx \
  --output-dir results/visualizations/ \
  --disease-id MONDO:0004979 \
  --disease-name "asthma"
```

#### 4. Run Complete Pipeline
```bash
biomedgps-explainer pipeline \
  --disease-id MONDO:0004979 \
  --output-dir results/ \
  --filter-expression "score > 0.6 and existing == False"
```

**Description**: Executes the complete workflow (run → filter → visualize) in a single command.

**Key Parameters**:
- `--disease-id`: Disease ID (required)
- `--output-dir`: Output directory (default: results)
- `--filter-expression`: Optional filter to apply to results
- `--model-run-id`: Model run ID (default: 6vlvgvfq)
- `--top-n-diseases`: Number of similar diseases (default: 100)
- `--threshold`: Drug filtering threshold (default: 0.5)
- `--top-n-drugs`: Number of drugs to interpret (default: 100)

### Python API

#### Basic Workflow
```python
from drugs4disease.core import DrugDiseaseCore
from drugs4disease.filter import DrugFilter
from drugs4disease.visualizer import Visualizer

# Initialize components
core = DrugDiseaseCore()
filter_tool = DrugFilter()
visualizer = Visualizer(disease_id="MONDO:0004979", disease_name="asthma")

# Run analysis
core.run_full_pipeline(
    disease_id="MONDO:0004979",
    output_dir="results/",
    top_n_diseases=50,
    top_n_drugs=100
)

# Filter results
filter_tool.filter_drugs(
    input_file="results/annotated_drugs.xlsx",
    expression="score > 0.7 and num_of_shared_genes_in_path >= 1",
    output_file="results/filtered_drugs.xlsx"
)

# Generate report
visualizer.generate_report(
    data_file="results/filtered_drugs.xlsx",
    output_file="results/analysis_report.html"
)
```

## Advanced Features

### Custom Filtering

The filtering system supports complex logical expressions:

#### Basic Filters
```python
# High-scoring drugs
"score > 0.8"

# New drug candidates
"existing == False"

# Drugs with shared genes
"num_of_shared_genes_in_path >= 2"
```

#### Complex Filters
```python
# High-scoring new drugs with biological evidence
"score > 0.7 and existing == False and num_of_shared_genes_in_path >= 1"

# Network-central drugs
"drug_degree > 10 and num_of_key_genes >= 3"

# Multiple criteria
"(score > 0.6 and existing == False) or (num_of_shared_pathways >= 2)"
```

### Custom Visualizations

Generate specific chart types:

```python
# Generate score distribution
visualizer.create_visualization(
    data_file="results/annotated_drugs.xlsx",
    viz_type="score_distribution",
    output_file="results/score_dist.png"
)

# Generate network analysis
visualizer.create_visualization(
    data_file="results/annotated_drugs.xlsx",
    viz_type="network_centrality",
    output_file="results/network_analysis.png"
)
```

### Available Chart Types

1. `score_distribution` - Predicted score distribution
2. `predicted_score_boxplot` - Score by knowledge graph inclusion
3. `disease_similarity_heatmap` - Drug similarity heatmap
4. `network_centrality` - Network centrality analysis
5. `shared_genes_pathways` - Gene/pathway overlap analysis
6. `drug_disease_network` - Drug similarity network
7. `shared_gene_count` - Shared gene count distribution
8. `score_vs_degree` - Score vs degree relationship
9. `shared_gene_count_vs_score` - Interactive gene overlap vs score
10. `shared_pathways` - Shared pathways distribution
11. `key_genes_distribution` - Key genes distribution
12. `existing_vs_predicted` - Known vs predicted drugs ratio

## Visualization Guide

### Understanding the Outputs

#### 1. Score Distribution
- **What it shows**: Distribution of predicted scores for all candidate drugs
- **How to interpret**: Higher scores indicate stronger predicted drug-disease associations
- **Key insights**: Look for drugs in the upper tail (score > 0.7)

#### 2. Network Centrality
- **What it shows**: How central drugs are in the biological network
- **How to interpret**: Higher centrality suggests stronger regulatory potential
- **Key insights**: Combine with scores for comprehensive evaluation

#### 3. Shared Genes Analysis
- **What it shows**: Number of genes shared between drugs and diseases
- **How to interpret**: More shared genes suggest stronger biological relevance
- **Key insights**: Look for drugs with multiple shared genes

#### 4. Pathway Overlap
- **What it shows**: Biological pathways affected by both drugs and diseases
- **How to interpret**: Overlapping pathways suggest mechanistic relevance
- **Key insights**: Drugs affecting disease-relevant pathways are promising

### Interactive HTML Report

The toolkit generates an interactive HTML report with:

- **Interactive charts**: Hover for details, zoom, pan
- **Sortable tables**: Click headers to sort drug lists
- **Filterable data**: Use search boxes to find specific drugs
- **Exportable results**: Download data in various formats

## Troubleshooting

### Common Issues and Solutions

#### 1. Missing Data Files
**Error**: `FileNotFoundError: Model file validation failed`

**Solution**:
```bash
# Check if files exist
python3 examples/run_data_validation.py

# Ensure correct directory structure
ls -la data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/
```

#### 2. Memory Issues
**Error**: `MemoryError` or slow performance

**Solution**:
```python
# Reduce analysis scope
core.run_full_pipeline(
    disease_id="MONDO:0004979",
    top_n_diseases=25,  # Reduce from 100
    top_n_drugs=50,     # Reduce from 1000
    output_dir="results/"
)
```

#### 3. Visualization Errors
**Error**: `Unable to determine Axes to steal space for Colorbar`

**Solution**: This has been fixed in recent versions. Update to the latest version:
```bash
pip install -e . --upgrade
```

#### 4. ZIP File Issues
**Error**: `zipfile.BadZipFile`

**Solution**:
```bash
# Check ZIP file integrity
unzip -t your_file.zip

# Re-download or re-compress the file
```

### Getting Help

#### Check CLI Help
```bash
biomedgps-explainer --help
biomedgps-explainer run --help
biomedgps-explainer filter --help
biomedgps-explainer visualize --help
biomedgps-explainer pipeline --help
```

#### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run your analysis
core.run_full_pipeline(...)
```

## Best Practices

### 1. Data Management
- **Backup your data**: Keep copies of original model files
- **Use version control**: Track changes to analysis scripts
- **Organize outputs**: Use descriptive directory names

### 2. Analysis Workflow
- **Start small**: Test with small datasets first
- **Validate inputs**: Always run data validation
- **Document parameters**: Keep track of analysis settings
- **Iterate**: Refine filters based on initial results

### 3. Performance Optimization
- **Use appropriate parameters**: Adjust `top_n_diseases` and `top_n_drugs`
- **Filter early**: Apply filters early in the pipeline
- **Monitor resources**: Watch memory and CPU usage
- **Use parallel processing**: For independent analyses

### 4. Result Interpretation
- **Consider biological context**: Don't rely solely on scores
- **Validate findings**: Cross-reference with literature
- **Document assumptions**: Note any assumptions made
- **Share results**: Use the HTML reports for presentations

### 5. Reproducibility
- **Set random seeds**: For reproducible results
- **Version dependencies**: Document package versions
- **Save intermediate results**: For debugging and validation
- **Use configuration files**: For complex parameter sets

## FAQ

### Q: What disease IDs are supported?
A: The toolkit supports any disease ID present in your knowledge graph. Common formats include:
- MONDO IDs: `MONDO:0004979` (asthma)
- DO IDs: `DOID:2841` (asthma)
- OMIM IDs: `OMIM:600807` (asthma)

### Q: How do I find disease IDs?
A: You can find disease IDs by:
1. Searching the MONDO database: https://monarchinitiative.org/
2. Using the OMIM database: https://omim.org/
3. Checking your entity file for available diseases

### Q: What do the scores mean?
A: Scores range from 0 to 1, where:
- 0.0-0.3: Low confidence prediction
- 0.3-0.6: Medium confidence prediction
- 0.6-0.8: High confidence prediction
- 0.8-1.0: Very high confidence prediction

### Q: How accurate are the predictions?
A: Accuracy depends on:
- Quality of the knowledge graph
- Completeness of drug-disease associations
- Model training parameters
- Disease-specific factors

### Q: Can I use my own knowledge graph?
A: Yes, but ensure it follows the required format:
- Entity file with id, label, name columns
- Knowledge graph with source/target information
- Embeddings in the correct format

### Q: How long does analysis take?
A: Analysis time depends on:
- Dataset size (drugs and diseases)
- Hardware specifications
- Analysis parameters
- Typically 10-60 minutes for standard analyses

### Q: Can I run multiple diseases at once?
A: Currently, the toolkit processes one disease at a time. You can:
1. Run multiple analyses in parallel
2. Create a batch script for multiple diseases
3. Combine results manually

### Q: How do I interpret the network analysis?
A: Network analysis shows:
- **Degree**: Number of connections (higher = more central)
- **Key genes**: Genes with high network centrality
- **Pathways**: Biological pathways affected by drugs

### Q: What's the difference between existing and predicted drugs?
A: - **Existing drugs**: Known to treat the disease (validation set)
- **Predicted drugs**: New candidates discovered by the model

### Q: How do I cite this toolkit?
A: Use the citation provided in the main README:
```bibtex
@software{biomedgps_explainer,
  title={BioMedGPS Explainer: A Knowledge Graph-Based Drug Discovery Toolkit},
  author={Yang, Jingcheng},
  year={2024},
  url={https://github.com/open-prophetdb/biomedgps-explainer}
}
```

## Additional Resources

- **API Documentation**: See `docs/API.md` for detailed API reference
- **Examples**: Check the `examples/` directory for working examples
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions for help and tips

## Support

For additional support:
- **Email**: yjcyxky@163.com
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the docs directory for detailed guides 