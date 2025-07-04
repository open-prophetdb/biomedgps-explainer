"""
Utility functions module
Contains ZIP decompression, file path handling, etc.
"""

import os
import zipfile
import glob
from typing import List, Tuple, Optional


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
    
    for file_path, description in files_to_check:
        if not os.path.exists(file_path):
            print(f"✗ {description} does not exist: {file_path}")
            return False
        else:
            print(f"✓ {description}: {file_path}")
    
    return True 