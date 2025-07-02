import click
import pandas as pd
import os
from .core import DrugDiseaseCore
from .filter import DrugFilter
from .visualizer import DrugVisualizer

@click.group()
def cli():
    """drugs4disease: 生成潜在药物列表、筛选与可视化。"""
    pass

@cli.command()
@click.option('--disease', required=True, help='疾病名称或ID')
@click.option('--entity-file', required=True, help='实体文件')
@click.option('--knowledge-graph', required=True, help='知识图谱文件')
@click.option('--entity-embeddings', required=True, help='实体嵌入文件')
@click.option('--relation-embeddings', required=True, help='关系嵌入文件')
@click.option('--output-dir', required=True, help='输出目录')
@click.option('--model', default='TransE_l2', help='KGE模型')
@click.option('--top-n-diseases', default=100, help='相似疾病数量')
@click.option('--gamma', default=12.0, help='Gamma参数')
@click.option('--threshold', default=0.5, help='药物筛选阈值')
@click.option('--relation-type', default='DGIDB::OTHER::Gene:Compound', help='关系类型')
def run(disease, entity_file, knowledge_graph, entity_embeddings, relation_embeddings, output_dir, model, top_n_diseases, gamma, threshold, relation_type):
    """一键完成所有分析步骤，生成 annotated_drugs.xlsx"""
    os.makedirs(output_dir, exist_ok=True)
    core = DrugDiseaseCore()
    
    try:
        core.run_full_pipeline(
            disease_id=disease,
            entity_file=entity_file,
            knowledge_graph=knowledge_graph,
            entity_embeddings=entity_embeddings,
            relation_embeddings=relation_embeddings,
            output_dir=output_dir,
            model=model,
            top_n_diseases=top_n_diseases,
            gamma=gamma,
            threshold=threshold,
            relation_type=relation_type
        )
        click.echo(f'✅ 分析完成！结果已保存到 {os.path.join(output_dir, "annotated_drugs.xlsx")}')
    except Exception as e:
        click.echo(f'❌ 分析失败: {str(e)}', err=True)
        raise

@cli.command()
@click.option('--expression', required=True, help='筛选表达式')
@click.option('--input', 'input_file', required=True, help='输入文件 annotated_drugs.xlsx')
@click.option('--output', 'output_file', required=True, help='输出文件 final_drugs.xlsx')
def filter(expression, input_file, output_file):
    """根据表达式筛选药物列表"""
    df = pd.read_excel(input_file)
    filtered = DrugFilter.filter_dataframe(df, expression)
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name='original', index=False)
        filtered.to_excel(writer, sheet_name='filtered', index=False)
    click.echo(f'已生成 {output_file}')

@cli.command()
@click.option('--input', 'input_file', required=True, help='输入文件 annotated_drugs.xlsx')
@click.option('--output-dir', required=True, help='图表输出目录')
@click.option('--viz-type', default='all', help='可视化类型 (all, score_distribution, top_drugs_bar, disease_similarity_heatmap, network_centrality, shared_genes_pathways, drug_disease_network)')
def visualize(input_file, output_dir, viz_type):
    """生成可视化图表和报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = DrugVisualizer()
    
    if viz_type == 'all':
        # 生成所有类型的可视化
        viz_types = [
            'score_distribution',
            'top_drugs_bar', 
            'disease_similarity_heatmap',
            'network_centrality',
            'shared_genes_pathways',
            'drug_disease_network'
        ]
        
        for vt in viz_types:
            try:
                output_file = os.path.join(output_dir, f"{vt}.png")
                visualizer.create_visualization(
                    data_file=input_file,
                    viz_type=vt,
                    output_file=output_file
                )
                click.echo(f'✓ 生成图表: {vt}')
            except Exception as e:
                click.echo(f'✗ 生成图表失败 {vt}: {e}')
        
        # 生成综合报告
        report_file = os.path.join(output_dir, "analysis_report.html")
        try:
            visualizer.generate_report(
                data_file=input_file,
                output_file=report_file,
                title="药物发现分析报告"
            )
            click.echo(f'✓ 生成综合报告: {report_file}')
        except Exception as e:
            click.echo(f'✗ 生成报告失败: {e}')
    else:
        # 生成单个类型的可视化
        output_file = os.path.join(output_dir, f"{viz_type}.png")
        try:
            visualizer.create_visualization(
                data_file=input_file,
                viz_type=viz_type,
                output_file=output_file
            )
            click.echo(f'✓ 生成图表: {output_file}')
        except Exception as e:
            click.echo(f'✗ 生成图表失败: {e}')
    
    click.echo(f'✅ 可视化完成！输出目录: {output_dir}')

if __name__ == '__main__':
    cli() 