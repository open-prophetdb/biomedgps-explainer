# Changelog

All notable changes to the BioMedGPS Explainer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive English documentation
- API documentation with detailed examples
- User guide with step-by-step instructions
- Troubleshooting section with common issues and solutions
- Best practices guide for optimal usage
- FAQ section addressing common questions

### Changed
- Updated README.md with comprehensive project overview
- Translated examples README to English
- Improved error messages and user feedback
- Enhanced documentation structure and organization

### Fixed
- Fixed colorbar creation issue in drug-disease network visualization
- Improved axes handling in matplotlib-based visualizations
- Enhanced error handling for missing data files

## [0.1.0] - 2024-01-XX

### Added
- Initial release of BioMedGPS Explainer toolkit
- Core drug discovery analysis pipeline
- Knowledge Graph Embedding (KGE) based drug prediction
- Network analysis and centrality calculations
- Pathway enrichment analysis
- Shared gene and disease annotation
- Advanced drug filtering with logical expressions
- Comprehensive visualization system with 12 chart types
- Interactive HTML report generation
- Command line interface (CLI) with multiple commands
- Python API for programmatic usage
- Automatic ZIP file decompression support
- Data validation and integrity checking
- AI-powered drug explanation generation

### Features
- **DrugDiseaseCore**: Main analysis engine for drug discovery
- **DrugFilter**: Advanced filtering with complex logical expressions
- **Visualizer**: Comprehensive visualization and reporting system
- **DrugExplain**: AI-powered drug explanation generation
- **CLI Tools**: Command-line interface for easy usage
- **Data Validation**: Built-in data validation and ZIP decompression

### Supported Visualizations
1. Score Distribution - Predicted score distribution of candidate drugs
2. Predicted Score Boxplot - Score distribution by knowledge graph inclusion
3. Disease Similarity Heatmap - Drug similarity based on shared diseases
4. Network Centrality - Drug network centrality analysis
5. Shared Genes and Pathways - Comprehensive gene/pathway overlap analysis
6. Drug Similarity Network - Interactive drug relationship network
7. Shared Gene Count - Distribution of shared genes between drugs and diseases
8. Score vs Degree - Relationship between network degree and predicted scores
9. Shared Gene Count vs Score - Interactive scatter plot of gene overlap vs scores
10. Shared Pathways - Distribution of overlapping pathways
11. Key Genes Distribution - Distribution of key genes in PPI networks
12. Existing vs Predicted - Ratio of known to predicted drugs

### Technical Specifications
- **Python Version**: 3.8+
- **Dependencies**: pandas, numpy, matplotlib, seaborn, networkx, gseapy, torch, plotly
- **Data Formats**: TSV, Excel, JSON
- **Output Formats**: Excel, HTML, PNG, JSON
- **Supported Models**: TransE_l2 (extensible to other KGE models)

### Data Requirements
- Entity annotations file (annotated_entities.tsv)
- Knowledge graph triples file (knowledge_graph.tsv)
- Entity embeddings file (entity_embeddings.tsv)
- Relation embeddings file (relation_type_embeddings.tsv)

### CLI Commands
- `drugs4disease run` - Run complete drug discovery analysis
- `drugs4disease filter` - Filter drug candidates
- `drugs4disease visualize` - Generate visualizations and reports

### Example Usage
```bash
# Run complete analysis
drugs4disease run --disease MONDO:0004979 --output-dir results/

# Filter results
drugs4disease filter --input-file results/annotated_drugs.xlsx --expression "score > 0.6" --output-file results/filtered_drugs.xlsx

# Generate visualizations
drugs4disease visualize --input-file results/filtered_drugs.xlsx --output-dir results/visualizations/
```

### Python API
```python
from drugs4disease.core import DrugDiseaseCore
from drugs4disease.filter import DrugFilter
from drugs4disease.visualizer import Visualizer

# Initialize and run analysis
core = DrugDiseaseCore()
core.run_full_pipeline(disease_id="MONDO:0004979", output_dir="results/")

# Filter and visualize
filter_tool = DrugFilter()
visualizer = Visualizer()
filter_tool.filter_drugs(input_file="results/annotated_drugs.xlsx", expression="score > 0.7", output_file="results/filtered_drugs.xlsx")
visualizer.generate_report(data_file="results/filtered_drugs.xlsx", output_file="results/analysis_report.html")
```

### Performance Characteristics
- **Memory Usage**: 8-16GB RAM recommended for large datasets
- **Processing Time**: 10-60 minutes for standard analyses
- **Scalability**: Supports datasets with thousands of drugs and diseases
- **Parallelization**: Supports parallel processing for independent analyses

### Quality Assurance
- Comprehensive test suite covering core functionality
- Data validation and integrity checking
- Error handling and recovery mechanisms
- Performance optimization for large datasets
- Documentation and examples for all features

### Community and Support
- Open-source MIT license
- Comprehensive documentation
- Example scripts and tutorials
- Community support through GitHub issues
- Regular updates and maintenance

---

## Version History Summary

### Major Versions
- **0.1.0**: Initial release with core functionality
- **Unreleased**: Documentation and bug fixes

### Key Milestones
- **Initial Development**: Core drug discovery pipeline
- **Feature Completion**: All major analysis components
- **Documentation**: Comprehensive English documentation
- **Bug Fixes**: Visualization and error handling improvements

### Future Roadmap
- Additional KGE model support
- Enhanced visualization options
- Performance optimizations
- Extended biological analysis capabilities
- Community-driven feature additions

---

## Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Documentation updates
- Issue reporting
- Feature requests

## Support

For support and questions:
- **Email**: yjcyxky@163.com
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the docs directory for detailed guides
- **Examples**: See the examples directory for working code samples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 