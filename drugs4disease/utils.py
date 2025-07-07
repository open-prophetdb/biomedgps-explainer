"""
Utility functions module
Contains ZIP decompression, file path handling, etc.
"""

import os
import zipfile
import glob
import pandas as pd
from typing import List, Tuple, Optional
import logging

def extract_zip_if_needed(zip_path: str, extract_dir: str) -> bool:
    """
    If the ZIP file exists, decompress it to the specified directory
    
    Args:
        zip_path: ZIP file path
        extract_dir: decompression target directory
        
    Returns:
        bool: whether the decompression is successful
    """
    if os.path.exists(zip_path):
        print(f"Found ZIP file: {zip_path}")
        print(f"Decompressing to: {extract_dir}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✓ Decompression completed")
            return True
        except Exception as e:
            print(f"✗ Decompression failed: {e}")
            return False
    return False


def extract_model_files_if_needed(model_dir: str) -> bool:
    """
    Check and decompress the ZIP file in the model directory
    
    Args:
        model_dir: model directory path
        
    Returns:
        bool: whether all files are successfully processed
    """
    print(f"检查模型目录: {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"Model directory does not exist: {model_dir}")
        return False
    
    # the list of files that need to be decompressed
    required_files = [
        "annotated_entities.tsv",
        "knowledge_graph.tsv", 
        "entity_embeddings.tsv",
        "relation_type_embeddings.tsv"
    ]
    
    # the corresponding ZIP files
    zip_files = [
        "annotated_entities.tsv.zip",
        "knowledge_graph.tsv.zip",
        "entity_embeddings.tsv.zip", 
        "relation_type_embeddings.tsv.zip"
    ]
    
    extracted_count = 0
    missing_files = []
    
    for i, (required_file, zip_file) in enumerate(zip(required_files, zip_files)):
        file_path = os.path.join(model_dir, required_file)
        zip_path = os.path.join(model_dir, zip_file)
        
        # if the target file does not exist but the ZIP file exists, decompress it
        if not os.path.exists(file_path) and os.path.exists(zip_path):
            print(f"\nDecompressing {zip_file}...")
            if extract_zip_if_needed(zip_path, model_dir):
                extracted_count += 1
            else:
                print(f"✗ Decompression of {zip_file} failed")
                missing_files.append(required_file)
        elif os.path.exists(file_path):
            print(f"✓ {required_file} already exists")
        else:
            print(f"✗ {required_file} does not exist, and the corresponding ZIP file also does not exist")
            missing_files.append(required_file)
    
    if extracted_count > 0:
        print(f"\n✓ Successfully decompressed {extracted_count} files")
    
    if missing_files:
        print(f"\n✗ Missing the following files: {missing_files}")
        return False
    
    return True


def get_default_model_dir():
    """
    Get the default model directory path
    
    Returns:
        str: default model directory path
    """
    return os.path.join(os.path.expanduser("~/.biomedgps-explainer/models"), "biomedgps_v2_20250318_TransE_l2_KMkgBhIV")


def _find_project_root():
    """
    Find the project root directory from the current working directory
    
    Returns:
        str: project root directory path
    """
    # first try to find the project root directory from the current file location
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = _search_project_root_from_dir(current_file_dir)
    
    if project_root is not None:
        return project_root
    
    # if not found from the current file location, try to find from the working directory
    cwd = os.getcwd()
    project_root = _search_project_root_from_dir(cwd)
    
    if project_root is not None:
        return project_root
    
    # if not found from the current file location and working directory, use the current working directory
    return cwd


def _search_project_root_from_dir(start_dir):
    """
    Search the project root directory from the specified directory
    
    Args:
        start_dir: the directory to start searching
        
    Returns:
        str or None: project root directory path, if not found, return None
    """
    current_dir = start_dir
    
    while current_dir != os.path.dirname(current_dir):  # until the root directory is reached
        # check if the project identifier file/directory exists
        indicators = [
            "data",  # data directory
            "setup.py",  # Python package setup file
            "pyproject.toml",  # modern Python project configuration
            ".git",  # Git repository
            "README.md",  # project description file
        ]
        
        for indicator in indicators:
            if os.path.exists(os.path.join(current_dir, indicator)):
                return current_dir
        
        current_dir = os.path.dirname(current_dir)
    
    return None


def get_model_file_paths(
    entity_file: Optional[str] = None,
    knowledge_graph: Optional[str] = None,
    entity_embeddings: Optional[str] = None,
    relation_embeddings: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    """
    Smartly get the model file paths
    
    逻辑：
    1. 如果用户指定了所有四个文件，使用用户指定的路径
    2. 如果用户没有指定任何文件，使用默认目录下的文件
    3. 如果用户指定了部分文件，抛出错误（要求四个文件必须一起使用）
    4. 自动处理ZIP文件的解压缩
    
    Args:
        entity_file: entity file path
        knowledge_graph: knowledge graph file path
        entity_embeddings: entity embeddings file path
        relation_embeddings: relation embeddings file path
        project_root: project root directory
        
    Returns:
        Tuple[str, str, str, str]: four model file paths
        
    Raises:
        ValueError: when the user only specifies some files
    """
    # check how many files the user specified
    specified_files = [
        (entity_file, "entity_file"),
        (knowledge_graph, "knowledge_graph"),
        (entity_embeddings, "entity_embeddings"),
        (relation_embeddings, "relation_embeddings")
    ]
    
    specified_count = sum(1 for file_path, _ in specified_files if file_path is not None)
    
    if specified_count == 0:
        # the user did not specify any file, use the default directory
        print("No model files specified, using the files in the default directory...")
        model_dir = get_default_model_dir()
        
        # check and decompress the ZIP file
        if not extract_model_files_if_needed(model_dir):
            raise FileNotFoundError(f"The files in the default model directory are incomplete: {model_dir}")
        
        # return the default file paths
        return (
            os.path.join(model_dir, "annotated_entities.tsv"),
            os.path.join(model_dir, "knowledge_graph.tsv"),
            os.path.join(model_dir, "entity_embeddings.tsv"),
            os.path.join(model_dir, "relation_type_embeddings.tsv")
        )
    
    elif specified_count == 4:
        # the user specified all four files
        print("Using the user-specified model files...")
        
        # check if each file exists, if not, but the corresponding ZIP file exists, decompress it
        file_paths = []
        for file_path, file_name in specified_files:
            if not os.path.exists(file_path):
                # check if there is a corresponding ZIP file
                zip_path = file_path + ".zip"
                if os.path.exists(zip_path):
                    print(f"Found ZIP file: {zip_path}")
                    extract_dir = os.path.dirname(file_path)
                    if extract_zip_if_needed(zip_path, extract_dir):
                        print(f"✓ Decompression completed: {file_path}")
                    else:
                        raise FileNotFoundError(f"Decompression failed: {zip_path}")
                else:
                    raise FileNotFoundError(f"文件不存在: {file_path}")
            
            file_paths.append(file_path)
        
        return tuple(file_paths)
    
    else:
        # the user only specified some files
        specified_names = [name for _, name in specified_files if _ is not None]
        missing_names = [name for _, name in specified_files if _ is None]
        
        raise ValueError(
            f"Model files must be specified together. You specified: {specified_names}, "
            f"but missing: {missing_names}. Please either specify all four files, "
            f"or do not specify any file (use the default directory)"
        )


def _validate_tsv_file(file_path: str, required_columns: List[str], min_rows: int = 1) -> bool:
    """
    Validate a TSV file for format, content, and structure
    
    Args:
        file_path: path to the TSV file
        required_columns: list of required column names
        min_rows: minimum number of rows (excluding header)
        
    Returns:
        bool: whether the file is valid
    """
    try:
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            print(f"✗ File is empty: {file_path}")
            return False
        
        # Try to read the file
        df = pd.read_csv(file_path, sep='\t', nrows=1000)  # Read first 1000 rows for validation
        
        # Check if file has any content
        if len(df) == 0:
            print(f"✗ File has no data rows: {file_path}")
            return False
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"✗ Missing required columns {missing_columns} in: {file_path}")
            return False
        
        # Check minimum rows
        if len(df) < min_rows:
            print(f"✗ File has fewer than {min_rows} rows: {file_path}")
            return False
        
        # Check for completely empty columns
        empty_columns = []
        for col in required_columns:
            if col in df.columns and df[col].isna().all():
                empty_columns.append(col)
        
        if empty_columns:
            print(f"✗ Columns are completely empty {empty_columns} in: {file_path}")
            return False
        
        return True
        
    except pd.errors.EmptyDataError:
        print(f"✗ File is empty or has no valid data: {file_path}")
        return False
    except pd.errors.ParserError as e:
        print(f"✗ File format error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading file {file_path}: {e}")
        return False


def validate_model_files(
    entity_file: str,
    knowledge_graph: str,
    entity_embeddings: str,
    relation_embeddings: str
) -> bool:
    """
    Validate if the model files exist and are formatted correctly
    
    Args:
        entity_file: entity file path
        knowledge_graph: knowledge graph file path
        entity_embeddings: entity embeddings file path
        relation_embeddings: relation embeddings file path
        
    Returns:
        bool: whether all files are valid
    """
    files_to_check = [
        (entity_file, "entity annotation file"),
        (knowledge_graph, "knowledge graph file"),
        (entity_embeddings, "entity embeddings file"),
        (relation_embeddings, "relation embeddings file")
    ]

    # First check if all files exist
    for file_path, description in files_to_check:
        if not os.path.exists(file_path):
            print(f"✗ {description} does not exist: {file_path}")
            return False
        else:
            print(f"✓ {description}: {file_path}")

    # Now validate file content and structure
    try:
        # Validate entity annotation file
        entity_columns = ['id', 'label', 'name']
        if not _validate_tsv_file(entity_file, entity_columns, min_rows=1):
            print(f"✗ Entity annotation file validation failed: {entity_file}")
            return False
        
        # Validate knowledge graph file
        kg_columns = ['source_id', 'source_type', 'source_name', 'target_id', 'target_type', 'target_name', 'relation_type']
        if not _validate_tsv_file(knowledge_graph, kg_columns, min_rows=1):
            print(f"✗ Knowledge graph file validation failed: {knowledge_graph}")
            return False
        
        # Validate entity embeddings file
        entity_emb_columns = ['embedding_id', 'entity_id', 'entity_type', 'entity_name', 'embedding']
        if not _validate_tsv_file(entity_embeddings, entity_emb_columns, min_rows=1):
            print(f"✗ Entity embeddings file validation failed: {entity_embeddings}")
            return False
        
        # Validate relation embeddings file
        relation_emb_columns = ['id', 'embedding']
        if not _validate_tsv_file(relation_embeddings, relation_emb_columns, min_rows=1):
            print(f"✗ Relation embeddings file validation failed: {relation_embeddings}")
            return False
        
        print("✓ All model files are valid")
        return True
        
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def init_logger(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """
    Initialize and return a logger.

    Args:
        log_level: Logging level (default: INFO)
        log_file: If provided, logs will also be written to this file
        logger_name: If provided, returns a named logger; otherwise, root logger

    Returns:
        Configured logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # 如果已有 handler，先清掉，避免重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 文件 handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
