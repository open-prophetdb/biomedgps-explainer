#!/usr/bin/env python3
"""
简单的数据文件验证脚本
验证BioMedGPS数据文件是否存在且格式正确
"""

import os
import sys

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"✗ {description}: {file_path} (文件不存在)")
        return False

def check_file_format(file_path, required_columns, description):
    """检查文件格式（仅检查前几行）"""
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取前几行来检查格式
            lines = [f.readline().strip() for _ in range(3)]
            if not lines[0]:
                print(f"✗ {description}: 文件为空")
                return False
            
            # 检查TSV格式
            headers = lines[0].split('\t')
            print(f"  - 列数: {len(headers)}")
            print(f"  - 前几列: {headers[:5]}")
            
            # 检查必需列
            missing_cols = [col for col in required_columns if col not in headers]
            if missing_cols:
                print(f"✗ {description}: 缺少必需列: {missing_cols}")
                return False
            else:
                print(f"✓ {description}: 格式正确")
                return True
                
    except Exception as e:
        print(f"✗ {description}: 读取文件时出错: {e}")
        return False

def main():
    """主函数"""
    # 获取项目根目录路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("BioMedGPS数据文件验证")
    print("=" * 50)
    
    # 设置数据目录 - 从examples目录向上查找
    data_dir = os.path.join(project_root, "data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV")
    
    # 检查数据文件
    files_to_check = [
        (os.path.join(data_dir, "annotated_entities.tsv"), 
         ["id", "name", "label"], 
         "实体注释文件"),
        (os.path.join(data_dir, "knowledge_graph.tsv"), 
         ["source_id", "source_type", "target_id", "target_type", "relation_type"], 
         "知识图谱文件"),
        (os.path.join(data_dir, "entity_embeddings.tsv"), 
         ["entity_id", "entity_type", "embedding"], 
         "实体嵌入文件"),
        (os.path.join(data_dir, "relation_type_embeddings.tsv"), 
         ["embedding"], 
         "关系嵌入文件")
    ]
    
    all_files_exist = True
    all_formats_correct = True
    
    for file_path, required_cols, description in files_to_check:
        print(f"\n检查 {description}:")
        exists = check_file_exists(file_path, description)
        if exists:
            format_ok = check_file_format(file_path, required_cols, description)
            all_formats_correct = all_formats_correct and format_ok
        else:
            all_files_exist = False
    
    print("\n" + "=" * 50)
    if all_files_exist and all_formats_correct:
        print("✓ 所有数据文件验证通过！")
        print("\n可以运行以下命令进行测试:")
        print("python3 examples/example_usage.py")
        return 0
    else:
        print("✗ 数据文件验证失败")
        if not all_files_exist:
            print("- 请确保已下载并解压BioMedGPS数据文件")
        if not all_formats_correct:
            print("- 请检查数据文件格式是否正确")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 