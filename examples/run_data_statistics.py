#!/usr/bin/env python3
"""
数据统计脚本：展示BioMedGPS数据文件的基本信息和统计
不需要安装额外依赖，仅用于演示数据文件的使用
支持自动解压缩ZIP格式的模型文件
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from drugs4disease.utils import extract_model_files_if_needed, get_default_model_dir

def main():
    """主函数：展示数据统计信息"""
    
    print("BioMedGPS数据统计信息")
    print("=" * 50)
    
    # 获取默认模型目录
    model_dir = get_default_model_dir()
    
    # 检查并解压缩模型目录下的ZIP文件
    if not extract_model_files_if_needed(model_dir):
        print("\n模型文件准备失败")
        print("请确保以下任一条件满足:")
        print("1. 模型目录存在且包含解压缩后的TSV文件")
        print("2. 模型目录存在且包含对应的ZIP文件")
        return 1
    
    entity_file = os.path.join(model_dir, "annotated_entities.tsv")
    knowledge_graph = os.path.join(model_dir, "knowledge_graph.tsv")
    entity_embeddings = os.path.join(model_dir, "entity_embeddings.tsv")
    relation_embeddings = os.path.join(model_dir, "relation_type_embeddings.tsv")
    
    # 检查文件是否存在
    for file_path in [entity_file, knowledge_graph, entity_embeddings, relation_embeddings]:
        if not os.path.exists(file_path):
            print(f"错误：文件不存在 {file_path}")
            return 1
    
    print("✓ 所有数据文件存在")
    
    # 演示数据文件的基本信息
    print("\n数据文件统计信息:")
    print("-" * 30)
    
    # 1. 实体文件信息
    try:
        with open(entity_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"实体文件行数: {len(lines)}")
            
            # 统计实体类型
            entity_types = {}
            for line in lines[1:]:  # 跳过标题行
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    entity_type = parts[2]  # label列
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            print("实体类型分布:")
            for entity_type, count in sorted(entity_types.items()):
                print(f"  - {entity_type}: {count}")
                
    except Exception as e:
        print(f"读取实体文件时出错: {e}")
    
    # 2. 知识图谱信息
    try:
        with open(knowledge_graph, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\n知识图谱三元组数量: {len(lines) - 1}")  # 减去标题行
            
            # 统计关系类型
            relation_types = {}
            for line in lines[1:]:  # 跳过标题行
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    relation_type = parts[0]  # relation_type列
                    relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
            
            print("关系类型分布 (前10个):")
            for relation_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {relation_type}: {count}")
                
    except Exception as e:
        print(f"读取知识图谱文件时出错: {e}")
    
    # 3. 嵌入文件信息
    try:
        with open(entity_embeddings, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\n实体嵌入向量数量: {len(lines) - 1}")
            
            # 检查嵌入向量维度
            if len(lines) > 1:
                embedding_line = lines[1].strip().split('\t')
                if len(embedding_line) >= 5:
                    embedding_str = embedding_line[4]  # embedding列
                    embedding_dim = len(embedding_str.split('|'))
                    print(f"嵌入向量维度: {embedding_dim}")
                    
    except Exception as e:
        print(f"读取实体嵌入文件时出错: {e}")
    
    # 4. 关系嵌入文件信息
    try:
        with open(relation_embeddings, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"关系嵌入向量数量: {len(lines) - 1}")
            
    except Exception as e:
        print(f"读取关系嵌入文件时出错: {e}")
    
    print("\n" + "=" * 50)
    print("数据统计完成！")
    print("\n下一步操作:")
    print("1. 验证文件格式: python3 examples/run_data_validation.py")
    print("2. 运行完整分析: python3 examples/run_full_example.py")
    print("3. 使用CLI工具: drugs4disease run --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 