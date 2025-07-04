import click
import sys
import pandas as pd
import os
from .core import DrugDiseaseCore
from .filter import DrugFilter
from .visualizer import Visualizer
from .full_pipeline import main as run_full_pipeline
from .model import Model
from .utils import init_logger

logger = init_logger()

@click.group()
def cli():
    """drugs4disease: generate potential drug list, filter and visualize."""
    pass

@cli.command()
@click.option('--disease-id', required=True, help='disease id, e.g. MONDO:0004979')
@click.option('--model-run-id', required=False, help='model run id, e.g. 6vlvgvfq. we can access `https://wandb.ai/yjcyxky/biomedgps-kge-v1?nw=nwuseryjcyxky` to get the model run id. If not specified, use the default model run id.', default='6vlvgvfq')
@click.option('--output-dir', required=True, help='output directory')
@click.option('--top-n-diseases', default=100, help='number of similar diseases')
@click.option('--threshold', default=0.5, help='drug filtering threshold')
@click.option('--relation-type', default='GNBR::T::Compound:Disease', help='A relation type which means the treatment relationship between a compound and a disease.')
@click.option('--top-n-drugs', default=1000, help='number of drugs to interpret')  # number of drugs to interpret, default 1000, if number of drugs to interpret is specified, then top_n_drugs is required
def run(disease_id, model_run_id, output_dir, top_n_diseases, threshold, relation_type, top_n_drugs):
    """
    One-click complete all analysis steps, generate annotated_drugs.xlsx
    
    Model file processing logic:
    - If no model file is specified, use the file in the default directory (supports ZIP automatic decompression)
    - If all four model files are specified, use the specified files
    - If only some files are specified, an error will be reported (all four files must be specified together)
    """
    os.makedirs(output_dir, exist_ok=True)
    core = DrugDiseaseCore()

    model = Model("biomedgps-kge-v1")
    converted_files = model.download_and_convert(model_run_id)
    model_config = model.load_model_config(converted_files.get("model_dir"))
    model_name = model_config.get("model_name", None)
    assert model_name is not None, "Model name is not found in model config"
    gamma = model_config.get("gamma", None)
    assert gamma is not None, "Gamma is not found in model config"

    try:
        core.run_full_pipeline(
            disease_id=disease_id,
            entity_file=converted_files["annotated_entities"],
            knowledge_graph=converted_files["knowledge_graph"],
            entity_embeddings=converted_files["entity_embeddings"],
            relation_embeddings=converted_files["relation_embeddings"],
            output_dir=output_dir,
            model=model_name,
            top_n_diseases=top_n_diseases,
            gamma=gamma,
            threshold=threshold,
            relation_type=relation_type,
            top_n_drugs=top_n_drugs
        )
        logger.info(f'✅ Analysis completed! Results saved to {os.path.join(output_dir, "annotated_drugs.xlsx")}')
    except Exception as e:
        logger.error(f'❌ Analysis failed: {str(e)}', err=True)
        raise

@cli.command()
@click.option('--expression', required=True, help='filter expression')
@click.option('--input-file', 'input_file', required=True, help='input file annotated_drugs.xlsx')
@click.option('--output-file', 'output_file', required=True, help='output file final_drugs.xlsx')
def filter(expression, input_file, output_file):
    """filter drugs list based on expression, such as 'pvalue < 0.05 and num_of_shared_genes > 10'"""
    df = pd.read_excel(input_file)
    filtered = DrugFilter.filter_dataframe(df, expression)
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer, sheet_name='original', index=False)
        filtered.to_excel(writer, sheet_name='filtered', index=False)
    logger.info(f'✅ Filter completed! Results saved to {output_file}')


@cli.command()
@click.option(
    "--input-file", "input_file", required=True, help="input file annotated_drugs.xlsx"
)
@click.option("--output-dir", required=True, help="chart output directory")
@click.option(
    "--viz-type",
    default="all",
    help=f'visualization type (all, {", ".join(Visualizer.get_chart_types())})',
)
@click.option("--disease-id", required=True, help="Disease id, e.g. MONDO:0004979")
@click.option("--disease-name", required=True, help="Disease name, e.g. 'Alzheimer's disease'")
def visualize(input_file, output_dir, viz_type, disease_id, disease_name):
    """generate visualization charts and report"""
    os.makedirs(output_dir, exist_ok=True)
    visualizer = Visualizer(disease_id=disease_id, disease_name=disease_name)

    if viz_type == 'all':        
        # generate comprehensive report
        report_file = os.path.join(output_dir, "analysis_report.html")
        try:
            visualizer.generate_report(
                data_file=input_file,
                output_file=report_file,
                title="Drug Discovery Analysis Report"
            )
            logger.info(f"✅ Generate comprehensive report: {report_file}")
        except Exception as e:
            logger.error(f"❌ Generate report failed: {e}")
    else:
        # generate single type of visualization
        output_file = os.path.join(output_dir, f"{viz_type}.png")
        try:
            visualizer.create_visualization(
                data_file=input_file,
                viz_type=viz_type,
                output_file=output_file
            )
            logger.info(f"✅ Generate chart: {output_file}")
        except Exception as e:
            logger.error(f"❌ Generate chart failed: {e}")

    logger.info(f"✅ Visualization completed! Output directory: {output_dir}")


@cli.command()
@click.option("--disease-id", required=True, help="Disease id, e.g. MONDO:0004979")
@click.option(
    "--model-run-id",
    required=False,
    help="Model run id, e.g. 6vlvgvfq. we can access `https://wandb.ai/yjcyxky/biomedgps-kge-v1?nw=nwuseryjcyxky` to get the model run id. If not specified, use the default model run id.",
    default="6vlvgvfq",
)
@click.option("--filter-expression", required=False, help="Filter expression, e.g. 'pvalue < 0.05 and num_of_shared_genes > 10'", default=None)
@click.option(
    "--output-dir", required=False, help="Output directory", default="results"
)
@click.option('--top-n-diseases', default=100, help='number of similar diseases')
@click.option('--threshold', default=0.5, help='drug filtering threshold')
@click.option('--relation-type', default='GNBR::T::Compound:Disease', help='A relation type which means the treatment relationship between a compound and a disease.')
@click.option('--top-n-drugs', default=100, help='number of drugs to interpret')  # number of drugs to interpret, default 100, if number of drugs to interpret is specified, then top_n_drugs is required
def pipeline(disease_id, model_run_id, filter_expression, output_dir, top_n_diseases, threshold, relation_type, top_n_drugs):
    """Run full pipeline, run --> filter --> visualize."""
    run_full_pipeline(
        disease_id=disease_id,
        model_run_id=model_run_id,
        output_dir=output_dir,
        filter_expression=filter_expression,
        top_n_diseases=top_n_diseases,
        threshold=threshold,
        relation_type=relation_type,
        top_n_drugs=top_n_drugs,
    )

if __name__ == '__main__':
    cli() 
