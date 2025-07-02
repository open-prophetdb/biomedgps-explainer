#!/usr/bin/env python3
"""
drugs4disease 使用示例

这个脚本展示了如何使用 drugs4disease 包进行药物发现分析。
"""

import os
import sys

# 添加项目根目录到Python路径，以便导入drugs4disease包
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from drugs4disease.core import DrugDiseaseCore
from drugs4disease.filter import DrugFilter
from drugs4disease.visualizer import DrugVisualizer

def main():
    """主函数：演示完整的药物发现工作流程"""
    
    # 初始化核心组件
    core = DrugDiseaseCore()
    drug_filter = DrugFilter()
    visualizer = DrugVisualizer()
    
    # 设置数据文件路径（BioMedGPS格式）- 从examples目录向上查找
    data_dir = os.path.join(project_root, "data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV")
    entity_file = os.path.join(data_dir, "annotated_entities.tsv")
    knowledge_graph = os.path.join(data_dir, "knowledge_graph.tsv")
    entity_embeddings = os.path.join(data_dir, "entity_embeddings.tsv")
    relation_embeddings = os.path.join(data_dir, "relation_type_embeddings.tsv")
    
    # 检查数据文件是否存在
    required_files = [entity_file, knowledge_graph, entity_embeddings, relation_embeddings]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误：数据文件不存在: {file_path}")
            print("请确保已下载并解压BioMedGPS数据文件到data目录")
            sys.exit(1)
    
    # 设置输出目录 - 在项目根目录下创建results
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例疾病ID（需要根据实际数据调整）
    # 从实体文件中找到一个疾病ID
    import pandas as pd
    try:
        entity_df = pd.read_csv(entity_file, sep="\t")
        disease_entities = entity_df[entity_df["label"] == "Disease"]
        if len(disease_entities) > 0:
            disease_id = disease_entities.iloc[0]["id"]
            print(f"使用疾病ID: {disease_id}")
        else:
            print("警告：未找到疾病实体，使用默认ID")
            disease_id = "MESH:D001249"  # 默认疾病ID
    except Exception as e:
        print(f"警告：无法读取实体文件，使用默认疾病ID: {e}")
        disease_id = "MESH:D001249"  # 默认疾病ID
    
    print("=" * 60)
    print("开始药物发现分析流程")
    print("=" * 60)
    
    try:
        # 步骤1：运行完整分析流程
        print("\n1. 运行完整分析流程...")
        core.run_full_pipeline(
            disease_id=disease_id,
            entity_file=entity_file,
            knowledge_graph=knowledge_graph,
            entity_embeddings=entity_embeddings,
            relation_embeddings=relation_embeddings,
            output_dir=output_dir,
            model='TransE_l2',
            top_n_diseases=50,
            gamma=12.0,
            threshold=0.5,
            relation_type='DGIDB::OTHER::Gene:Compound'
        )
        
        annotated_file = os.path.join(output_dir, "annotated_drugs.xlsx")
        if os.path.exists(annotated_file):
            print(f"✓ 完整分析完成，结果保存在: {annotated_file}")
        else:
            print("✗ 完整分析失败")
            return
        
        # 步骤2：应用复杂过滤表达式
        print("\n2. 应用过滤条件...")
        # 示例过滤表达式：选择分数大于0.6且治疗相似疾病数量大于2的药物
        filter_expression = "score > 0.6 and num_of_treated_similar_diseases > 2"
        
        filtered_file = os.path.join(output_dir, "filtered_drugs.xlsx")
        drug_filter.filter_drugs(
            input_file=annotated_file,
            expression=filter_expression,
            output_file=filtered_file
        )
        
        if os.path.exists(filtered_file):
            print(f"✓ 过滤完成，结果保存在: {filtered_file}")
        else:
            print("✗ 过滤失败")
            return
        
        # 步骤3：生成可视化报告
        print("\n3. 生成可视化报告...")
        report_dir = os.path.join(output_dir, "visualization_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成多种可视化图表
        visualizations = [
            "score_distribution",
            "top_drugs_bar",
            "disease_similarity_heatmap",
            "network_centrality",
            "shared_genes_pathways",
            "drug_disease_network"
        ]
        
        for viz_type in visualizations:
            try:
                output_file = os.path.join(report_dir, f"{viz_type}.png")
                visualizer.create_visualization(
                    data_file=filtered_file,
                    viz_type=viz_type,
                    output_file=output_file
                )
                print(f"✓ 生成图表: {viz_type}")
            except Exception as e:
                print(f"✗ 生成图表失败 {viz_type}: {e}")
        
        # 生成综合报告
        report_file = os.path.join(report_dir, "analysis_report.html")
        try:
            visualizer.generate_report(
                data_file=filtered_file,
                output_file=report_file,
                title=f"药物发现分析报告 - {disease_id}"
            )
            print(f"✓ 生成综合报告: {report_file}")
        except Exception as e:
            print(f"✗ 生成报告失败: {e}")
        
        print("\n" + "=" * 60)
        print("分析流程完成！")
        print("=" * 60)
        print(f"主要结果文件:")
        print(f"  - 完整分析结果: {annotated_file}")
        print(f"  - 过滤后结果: {filtered_file}")
        print(f"  - 可视化报告: {report_dir}/")
        print(f"  - 综合报告: {report_file}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 