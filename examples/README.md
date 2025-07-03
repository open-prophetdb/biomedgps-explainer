# Example Scripts

This directory contains example scripts for using the drugs4disease package.

## Script Descriptions

### run_data_validation.py
**Purpose**: Validate the existence and format of BioMedGPS data files, with support for automatic ZIP file decompression
**Dependencies**: No additional dependencies, uses only Python standard library
**Run**: `python3 examples/run_data_validation.py`

Features:
- Automatically detect and decompress ZIP format model files
- Validate the following files:
  - `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/annotated_entities.tsv`
  - `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/knowledge_graph.tsv`
  - `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/entity_embeddings.tsv`
  - `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/relation_type_embeddings.tsv`

### run_data_statistics.py
**Purpose**: Display basic information and statistics of BioMedGPS data files, with support for automatic ZIP file decompression
**Dependencies**: No additional dependencies, uses only Python standard library
**Run**: `python3 examples/run_data_statistics.py`

Features:
- Automatically detect and decompress ZIP format model files
- Statistics on entity type distribution
- Statistics on relation type distribution
- Display embedding vector dimensions
- Provide data file overview

### run_full_example.py
**Purpose**: Complete drug discovery analysis workflow demonstration, with support for automatic ZIP file decompression
**Dependencies**: Requires installation of drugs4disease package and its dependencies
**Run**: `python3 examples/run_full_example.py`

Complete workflow:
1. Automatically detect and decompress ZIP format model files
2. Run drug prediction analysis
3. Apply filtering conditions
4. Generate visualization reports
5. Output comprehensive analysis report

## Automatic Decompression Feature

All scripts support automatic detection and decompression of ZIP format model files:

### Supported ZIP File Formats
Scripts automatically detect the following ZIP files in the model directory:
- `annotated_entities.tsv.zip` - Entity annotation information
- `knowledge_graph.tsv.zip` - Knowledge graph triples
- `entity_embeddings.tsv.zip` - Entity embedding vectors
- `relation_type_embeddings.tsv.zip` - Relation type embedding vectors

### Decompression Logic
1. Check if the model directory exists
2. For each required TSV file, check if the corresponding ZIP file exists
3. If the TSV file doesn't exist but the ZIP file exists, automatically decompress
4. Continue normal workflow after decompression is complete

### Usage Scenarios
- **ZIP Format**: Place model files in ZIP format in the model directory, scripts will automatically decompress
- **Already Decompressed**: If TSV files already exist, scripts will use them directly, skipping decompression
- **Mixed Format**: Supports partial files in ZIP format, partial files already decompressed

## Usage Order

Recommended order for running examples:

1. **First validate data files**:
   ```bash
   python3 examples/run_data_validation.py
   ```

2. **View data statistics**:
   ```bash
   python3 examples/run_data_statistics.py
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

4. **Run complete example**:
   ```bash
   python3 examples/run_full_example.py
   ```

## Data File Preparation

### Method 1: Using ZIP Compressed Files (Recommended)
1. Place BioMedGPS model ZIP files in the model directory:
   ```
   data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/
   ├── annotated_entities.tsv.zip
   ├── knowledge_graph.tsv.zip
   ├── entity_embeddings.tsv.zip
   └── relation_type_embeddings.tsv.zip
   ```
2. Run example scripts, they will automatically decompress

### Method 2: Using Decompressed TSV Files
1. Manually decompress model files to `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/` directory
2. Ensure the following files are included:
   - `annotated_entities.tsv`
   - `knowledge_graph.tsv`
   - `entity_embeddings.tsv`
   - `relation_type_embeddings.tsv`

## Output Files

After running the complete example, the following files will be generated in the `results/` folder in the project root directory:
- `annotated_drugs.xlsx`: Complete annotated drug list
- `filtered_drugs.xlsx`: Filtered drug list
- `visualization_report/`: Visualization charts and reports

## Important Notes

- All scripts automatically search upward from the examples directory to find the project root directory
- Data file paths are automatically adjusted to be relative to the project root directory
- Output files are saved in the results folder in the project root directory
- Supports automatic decompression of ZIP format model files
- If data files don't exist, scripts provide clear error messages and usage guidance 