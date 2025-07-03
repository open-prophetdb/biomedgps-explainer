import click
import pandas as pd
import os
from .core import DrugDiseaseCore
from .filter import DrugFilter
from .visualizer import Visualizer
from .full_pipeline import main as run_full_pipeline

@click.group()
def cli():
    """drugs4disease: generate potential drug list, filter and visualize."""
    pass

@cli.command()
@click.option('--disease', required=True, help='disease id, e.g. MONDO:0004979')
@click.option('--entity-file', help='entity file (optional, use default directory if not specified)')
@click.option('--knowledge-graph', help='knowledge graph file (optional, use default directory if not specified)')
@click.option('--entity-embeddings', help='entity embeddings file (optional, use default directory if not specified)')
@click.option('--relation-embeddings', help='relation embeddings file (optional, use default directory if not specified)')
@click.option('--output-dir', required=True, help='output directory')
@click.option('--model', default='TransE_l2', help='KGE model')
@click.option('--top-n-diseases', default=100, help='number of similar diseases')
@click.option('--gamma', default=12.0, help='Gamma parameter')
@click.option('--threshold', default=0.5, help='drug filtering threshold')
@click.option('--relation-type', default='DGIDB::OTHER::Gene:Compound', help='relation type')
@click.option('--top-n-drugs', default=1000, help='number of drugs to interpret')  # number of drugs to interpret, default 1000, if number of drugs to interpret is specified, then top_n_drugs is required
def run(disease, entity_file, knowledge_graph, entity_embeddings, relation_embeddings, output_dir, model, top_n_diseases, gamma, threshold, relation_type, top_n_drugs):
    """
    One-click complete all analysis steps, generate annotated_drugs.xlsx
    
    Model file processing logic:
    - If no model file is specified, use the file in the default directory (supports ZIP automatic decompression)
    - If all four model files are specified, use the specified files
    - If only some files are specified, an error will be reported (all four files must be specified together)
    """
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
            relation_type=relation_type,
            top_n_drugs=top_n_drugs
        )
        click.echo(f'✅ Analysis completed! Results saved to {os.path.join(output_dir, "annotated_drugs.xlsx")}')
    except Exception as e:
        click.echo(f'❌ Analysis failed: {str(e)}', err=True)
        raise

@cli.command()
@click.option('--expression', required=True, help='filter expression')
@click.option('--input', 'input_file', required=True, help='input file annotated_drugs.xlsx')
@click.option('--output', 'output_file', required=True, help='output file final_drugs.xlsx')
def filter(expression, input_file, output_file):
    """filter drugs list based on expression"""
    df = pd.read_excel(input_file)
    filtered = DrugFilter.filter_dataframe(df, expression)
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name='original', index=False)
        filtered.to_excel(writer, sheet_name='filtered', index=False)
    click.echo(f'✅ Filter completed! Results saved to {output_file}')

@cli.command()
@click.option('--input', 'input_file', required=True, help='input file annotated_drugs.xlsx')
@click.option('--output-dir', required=True, help='chart output directory')
@click.option('--viz-type', default='all', help='visualization type (all, score_distribution, disease_similarity_heatmap, network_centrality, shared_genes_pathways, drug_disease_network, shared_gene_count, score_vs_degree, shared_pathways, key_genes_distribution, existing_vs_predicted)')
def visualize(input_file, output_dir, viz_type):
    """generate visualization charts and report"""
    os.makedirs(output_dir, exist_ok=True)

    visualizer = Visualizer()

    if viz_type == 'all':        
        # generate comprehensive report
        report_file = os.path.join(output_dir, "analysis_report.html")
        try:
            visualizer.generate_report(
                data_file=input_file,
                output_file=report_file,
                title="Drug Discovery Analysis Report"
            )
            click.echo(f"✅ Generate comprehensive report: {report_file}")
        except Exception as e:
            click.echo(f"❌ Generate report failed: {e}")
    else:
        # generate single type of visualization
        output_file = os.path.join(output_dir, f"{viz_type}.png")
        try:
            visualizer.create_visualization(
                data_file=input_file,
                viz_type=viz_type,
                output_file=output_file
            )
            click.echo(f"✅ Generate chart: {output_file}")
        except Exception as e:
            click.echo(f"❌ Generate chart failed: {e}")

    click.echo(f"✅ Visualization completed! Output directory: {output_dir}")
    
@cli.command()
@click.option('--disease-id', required=True, help='disease id, e.g. MONDO:0004979')
def run_full_pipeline(disease_id):
    """run full pipeline"""
    run_full_pipeline(disease_id=disease_id)

if __name__ == '__main__':
    cli() 
