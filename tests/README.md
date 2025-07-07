# Tests Directory

This directory contains comprehensive unit tests for the `drugs4disease` package. The tests are designed to validate the functionality of all major components and ensure code quality.

## Test Structure

### Core Module Tests (`test_core.py`)
Tests for the main `DrugDiseaseCore` class functionality:
- **Drug Prediction**: Tests the drug prediction pipeline with mock data
- **Shared Genes/Pathways Annotation**: Tests annotation of shared biological features
- **Shared Diseases Annotation**: Tests annotation of shared diseases
- **Network Features Annotation**: Tests network centrality and key gene analysis
- **Annotation Merging**: Tests the final annotation merging process
- **Full Pipeline**: Tests the complete end-to-end pipeline

### Filter Module Tests (`test_filter.py`)
Tests for the `DrugFilter` class:
- **Expression Parsing**: Tests parsing of filter expressions (&&, || operators)
- **DataFrame Filtering**: Tests filtering functionality with various conditions
- **File Operations**: Tests reading/writing Excel files with filtering
- **Error Handling**: Tests handling of invalid expressions and missing files

### Model Module Tests (`test_model.py`)
Tests for the `Model` class:
- **Wandb Integration**: Tests wandb login and API interactions
- **Model Download**: Tests downloading models from wandb
- **File Conversion**: Tests conversion of model files to required formats
- **Configuration Loading**: Tests loading model configuration files
- **File Validation**: Tests validation of numpy and TSV files

### Visualizer Module Tests (`test_visualizer.py`)
Tests for the `Visualizer` class:
- **Chart Generation**: Tests all visualization chart types
- **Report Generation**: Tests HTML report generation
- **Image Embedding**: Tests both embedded and file-based image handling
- **Data Handling**: Tests visualization with various data scenarios

### Utils Module Tests (`test_utils.py`)
Tests for utility functions:
- **ZIP Extraction**: Tests automatic ZIP file decompression
- **File Path Handling**: Tests smart model file path resolution
- **Project Root Detection**: Tests automatic project root finding
- **File Validation**: Tests model file validation
- **Logging**: Tests logger initialization

### Explain Module Tests (`test_explain.py`)
Tests for the `DrugExplain` class:
- **Prompt Generation**: Tests ChatGPT prompt generation
- **Template Rendering**: Tests Jinja2 template processing
- **Error Handling**: Tests handling of missing templates and invalid data

### Full Pipeline Tests (`test_full_pipeline.py`)
Tests for the complete pipeline:
- **End-to-End Execution**: Tests the complete workflow
- **Parameter Handling**: Tests default and custom parameters
- **Error Scenarios**: Tests error handling and validation failures
- **Integration**: Tests integration between all components

## Running Tests

### Run All Tests
```bash
# From the project root directory
python tests/run_tests.py

# Or using unittest directly
python -m unittest discover tests -v
```

### Run Specific Test Module
```bash
# Run only core tests
python tests/run_tests.py test_core

# Run only filter tests
python tests/run_tests.py test_filter

# Run only visualizer tests
python tests/run_tests.py test_visualizer
```

### Run Individual Test Files
```bash
# Run specific test file
python -m unittest tests.test_core -v
python -m unittest tests.test_filter -v
python -m unittest tests.test_model -v
python -m unittest tests.test_visualizer -v
python -m unittest tests.test_utils -v
python -m unittest tests.test_explain -v
python -m unittest tests.test_full_pipeline -v
```

### Run Specific Test Methods
```bash
# Run specific test method
python -m unittest tests.test_core.TestDrugDiseaseCore.test_predict_drugs -v
python -m unittest tests.test_filter.TestDrugFilter.test_parse_expression -v
```

## Test Data

The tests use mock data to avoid dependencies on external files:
- **Mock Entity Files**: Simulated annotated entities with diseases, drugs, and genes
- **Mock Knowledge Graph**: Simulated knowledge graph relationships
- **Mock Embeddings**: Simulated entity and relation embeddings
- **Mock Excel Files**: Simulated output files for testing

## Test Coverage

The tests cover:
- ✅ **Core Functionality**: All major methods in DrugDiseaseCore
- ✅ **Data Processing**: File reading, writing, and data manipulation
- ✅ **Error Handling**: Invalid inputs, missing files, and edge cases
- ✅ **Integration**: Component interactions and full pipeline
- ✅ **File Operations**: Excel, TSV, and ZIP file handling
- ✅ **Visualization**: Chart generation and report creation
- ✅ **Filtering**: Expression parsing and data filtering
- ✅ **Model Management**: Wandb integration and file conversion

## Mocking Strategy

The tests use extensive mocking to:
- **Isolate Components**: Test each component independently
- **Avoid External Dependencies**: Mock wandb, file system, and network calls
- **Control Test Data**: Use predictable mock data for consistent results
- **Speed Up Tests**: Avoid slow operations like file I/O and network calls

## Test Dependencies

The tests require the following packages:
- `unittest` (built-in)
- `pandas` (for data manipulation)
- `numpy` (for numerical operations)
- `matplotlib` (for visualization tests)
- `seaborn` (for visualization tests)
- `openpyxl` (for Excel file operations)

## Continuous Integration

These tests are designed to run in CI/CD environments:
- **Fast Execution**: Tests complete in under 30 seconds
- **No External Dependencies**: All external services are mocked
- **Deterministic Results**: Tests produce consistent results
- **Clear Error Messages**: Tests provide helpful error information

## Adding New Tests

When adding new functionality:
1. **Create Test File**: Add `test_<module_name>.py` for new modules
2. **Follow Naming Convention**: Use `Test<ClassName>` for test classes
3. **Use Descriptive Names**: Test methods should clearly describe what they test
4. **Include Edge Cases**: Test error conditions and boundary cases
5. **Mock External Dependencies**: Avoid real file system or network calls
6. **Add Documentation**: Include docstrings explaining test purpose

## Test Maintenance

Regular maintenance tasks:
- **Update Mock Data**: Keep mock data consistent with actual data formats
- **Review Test Coverage**: Ensure new functionality is tested
- **Update Dependencies**: Keep test dependencies up to date
- **Performance Monitoring**: Ensure tests remain fast and efficient 