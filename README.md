# BioMedGPS Explainer [中文文档](./README_zh.md) | [English Documentation](./README.md)

A comprehensive network medicine-based drug repurposing and visualization toolkit for biomedical research.

## Overview

BioMedGPS Explainer is a powerful Python toolkit designed for drug repurposing analysis using Knowledge Graph Embedding (KGE) models. It provides end-to-end capabilities for predicting potential drugs for specific diseases, analyzing drug-disease-gene relationships, and generating comprehensive visualization reports.

## Key Features

- **🔬 Drug Prediction**: Predict potential drugs using Knowledge Graph Embedding (KGE) models
- **🌐 Network Analysis**: Comprehensive drug-disease-gene pathway analysis with centrality calculations and PPI network analysis
- **🛤️ Pathway Enrichment**: Advanced pathway overlap analysis between drugs and diseases
- **🧬 Shared Annotations**: Statistical analysis of shared genes and diseases between drugs and diseases
- **🔍 Smart Filtering**: Advanced drug filtering with support for complex logical expressions
- **📊 Visualization**: Automatic chart generation with interactive plots and comprehensive HTML reports
- **📈 Data Validation**: Built-in data validation and automatic ZIP file decompression
- **🎯 Explainability**: AI-powered drug explanation generation for research insights

## Demo

[Drug Repurposing Analysis Report for Asthma](docs/reports/asthma_analysis_report.html)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd biomedgps-explainer

# Install the package in development mode
pip install -e .
```

### Install Dependencies

The package will automatically install all required dependencies:

- `click>=8.0.0` - Command line interface
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `matplotlib>=3.5.0` - Basic plotting
- `seaborn>=0.11.0` - Statistical visualization
- `networkx>=2.6.0` - Network analysis
- `gseapy>=0.12.0` - Gene set enrichment analysis
- `gprofiler-official>=1.0.0` - Functional profiling
- `torch>=1.9.0` - Deep learning framework
- `openpyxl>=3.0.0` - Excel file handling
- `requests>=2.25.0` - HTTP requests
- `scikit-learn>=1.0.0` - Machine learning utilities
- `plotly>=5.0.0` - Interactive visualizations
- `wandb>=0.15.0` - Weights & Biases integration

## Quick Start

### 1. Model Preparation

The toolkit automatically downloads pre-trained BioMedGPS model files from Weights & Biases (wandb) when you run the analysis. No manual model file preparation is required!

### 2. Validate Data Files

```bash
# Validate your data files
python3 examples/run_data_validation.py
```

### 3. Run Complete Analysis

```bash
# Run the complete drug discovery pipeline
python3 examples/run_full_example.py
```

### 4. Use Command Line Interface

The toolkit provides several CLI commands for different workflows:

#### Available Commands

- **`run`** - Execute drug discovery analysis only
- **`filter`** - Filter existing results based on criteria
- **`visualize`** - Generate charts and reports from results
- **`pipeline`** - Execute complete workflow (run → filter → visualize)

```bash
# Run analysis with custom parameters
biomedgps-explainer run --disease-id MONDO:0004979 --output-dir results/

# Generate visualizations
biomedgps-explainer visualize --input-file results/annotated_drugs.xlsx --output-dir results/visualizations/ --disease-id MONDO:0004979 --disease-name "asthma"

# Filter drugs
biomedgps-explainer filter --input-file results/annotated_drugs.xlsx --expression "score > 0.6 and existing == False" --output-file results/filtered_drugs.xlsx

# Run full pipeline (run → filter → visualize)
biomedgps-explainer pipeline --disease-id MONDO:0004979 --output-dir results/ --filter-expression "score > 0.6 and existing == False"
```

## Usage Examples

### Python API

```python
from drugs4disease.core import DrugDiseaseCore
from drugs4disease.filter import DrugFilter
from drugs4disease.visualizer import Visualizer

# Initialize components
core = DrugDiseaseCore()
drug_filter = DrugFilter()
visualizer = Visualizer(disease_id="MONDO:0004979", disease_name="asthma")

# Run complete analysis pipeline
core.run_full_pipeline(
    disease_id="MONDO:0004979",  # Disease ID
    output_dir="results/",
    model='TransE_l2',
    top_n_diseases=50,
    gamma=12.0,
    threshold=0.5,
    top_n_drugs=100
)

# Apply filtering
filter_expression = "pvalue < 0.05 and num_of_shared_genes_in_path >= 1 and existing == False"
drug_filter.filter_drugs(
    input_file="results/annotated_drugs.xlsx",
    expression=filter_expression,
    output_file="results/filtered_drugs.xlsx"
)

# Generate visualization report
visualizer.generate_report(
    data_file="results/filtered_drugs.xlsx",
    output_file="results/analysis_report.html",
    title="Drug Discovery Analysis Report"
)
```

### Advanced Filtering

The toolkit supports complex logical expressions for drug filtering:

```python
# Filter expressions examples
expressions = [
    # High-scoring new drugs
    "score > 0.7 and existing == False",
    
    # Drugs with shared genes and pathways
    "num_of_shared_genes_in_path >= 2 and num_of_shared_pathways >= 1",
    
    # Network-central drugs
    "drug_degree > 10 and num_of_key_genes >= 3",
    
    # Complex combination
    "score > 0.6 and existing == False and (num_of_shared_genes_in_path >= 1 or num_of_shared_pathways >= 1)"
]
```

## Output Files

The toolkit generates comprehensive output files:

### Main Results
- `annotated_drugs.xlsx` - Complete drug analysis with all annotations
- `filtered_drugs.xlsx` - Filtered drug candidates based on criteria

### Visualization Reports
- `analysis_report.html` - Interactive HTML report with all visualizations
- Individual chart files (PNG/JSON) for each analysis type

### Analysis Components
- `predicted_drugs.xlsx` - Initial drug predictions
- `shared_genes_pathways.xlsx` - Gene and pathway overlap analysis
- `shared_diseases.xlsx` - Disease similarity analysis
- `network_annotations.xlsx` - Network centrality features

## Visualization Types

The toolkit generates 12 different types of visualizations:

1. **Score Distribution** - Predicted score distribution of candidate drugs
2. **Predicted Score Boxplot** - Score distribution by knowledge graph inclusion
3. **Disease Similarity Heatmap** - Drug similarity based on shared diseases
4. **Network Centrality** - Drug network centrality analysis
5. **Shared Genes and Pathways** - Comprehensive gene/pathway overlap analysis
6. **Drug Similarity Network** - Interactive drug relationship network
7. **Shared Gene Count** - Distribution of shared genes between drugs and diseases
8. **Score vs Degree** - Relationship between network degree and predicted scores
9. **Shared Gene Count vs Score** - Interactive scatter plot of gene overlap vs scores
10. **Shared Pathways** - Distribution of overlapping pathways
11. **Key Genes Distribution** - Distribution of key genes in PPI networks
12. **Existing vs Predicted** - Ratio of known to predicted drugs

## Data Format

### Input Data Structure

The toolkit expects BioMedGPS format data files:

```
data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/
├── annotated_entities.tsv      # Entity annotations
├── knowledge_graph.tsv         # Knowledge graph triples
├── entity_embeddings.tsv       # Entity embeddings
└── relation_type_embeddings.tsv # Relation embeddings
```

### Entity File Format
```tsv
id  label   name
MONDO:0004979  Disease  asthma
CHEBI:12345    Compound aspirin
```

### Knowledge Graph Format
```tsv
source_id  source_type  source_name  target_id  target_type  target_name  relation_type
CHEBI:12345  Compound  aspirin  MONDO:0004979  Disease  asthma  GNBR::T::Compound:Disease
```

## Configuration

### Model Parameters

- `model`: KGE model type (default: 'TransE_l2')
- `gamma`: Margin parameter for KGE training (default: 12.0)
- `threshold`: Drug filtering threshold (default: 0.5)
- `top_n_diseases`: Number of similar diseases to consider (default: 100)
- `top_n_drugs`: Number of drugs to analyze (default: 1000)

### Filtering Options

The filtering system supports:
- Numerical comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=`)
- Logical operators (`and`, `or`, `not`)
- Boolean fields (`existing`, `is_key_gene`)
- String matching and pattern matching

## Troubleshooting

### Common Issues

1. **Missing Data Files**
   ```bash
   # Check if files exist
   python3 examples/run_data_validation.py
   ```

2. **ZIP File Decompression**
   - The toolkit automatically handles ZIP file decompression
   - Ensure ZIP files are in the correct directory structure

3. **Memory Issues**
   - Reduce `top_n_drugs` parameter for large datasets
   - Use smaller `top_n_diseases` values

4. **Visualization Errors**
   - Ensure all required plotting libraries are installed
   - Check file permissions for output directories

### Getting Help

```bash
# Check CLI help
biomedgps-explainer --help
biomedgps-explainer run --help
biomedgps-explainer filter --help
biomedgps-explainer visualize --help
biomedgps-explainer pipeline --help
```

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Development Setup

```bash
# Clone and setup development environment
git clone <repository-url>
cd biomedgps-explainer
pip install -e .
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

## Citation

If you use BioMedGPS Explainer in your research, please cite:

```bibtex
@software{biomedgps_explainer,
  title={BioMedGPS Explainer: A Network Medicine-Based Drug Discovery Toolkit},
  author={Yang, Jingcheng},
  year={2024},
  url={https://github.com/open-prophetdb/biomedgps-explainer}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Jingcheng Yang
- **Email**: yjcyxky@163.com
- **Project**: [BioMedGPS Explainer](https://github.com/open-prophetdb/biomedgps-explainer)

## Acknowledgments

- BioMedGPS team for the knowledge graph embeddings
- Open-source community for the underlying libraries
- Research community for feedback and contributions
