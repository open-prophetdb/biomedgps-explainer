#!/usr/bin/env python3
"""
drugs4disease example

This script demonstrates how to use the drugs4disease package for drug discovery analysis.
Support automatic decompression of ZIP format model files
"""

import os
import sys
import pandas as pd

# add project root directory to Python path, so that the drugs4disease package can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from drugs4disease.core import DrugDiseaseCore
from drugs4disease.filter import DrugFilter
from drugs4disease.visualizer import Visualizer
from drugs4disease.utils import get_model_file_paths, validate_model_files
from drugs4disease.explain import DrugExplain
from drugs4disease.model import Model


def main(disease_id: str, model_run_id: str = "6vlvgvfq"):
    """
    Main function: demonstrate the complete drug discovery workflow
    """

    # initialize the core components
    core = DrugDiseaseCore()
    drug_filter = DrugFilter()
    visualizer = Visualizer()

    try:
        model = Model("biomedgps-kge-v1")
        converted_files = model.download_and_convert(model_run_id)
        model_config = model.load_model_config(converted_files.get("model_dir"))
        model_name = model_config.get("model", None)
        assert model_name is not None, "Model name is not found in model config"
        gamma = model_config.get("gamma", None)
        assert gamma is not None, "Gamma is not found in model config"
    except Exception as e:
        sys.exit(1)

    # # smart get model file paths (support ZIP automatic decompression)
    # try:
    #     entity_file, knowledge_graph, entity_embeddings, relation_embeddings = get_model_file_paths()
    #     print("✅ Model file paths obtained successfully")
    # except Exception as e:
    #     print(f"❌ Model file paths obtained failed: {e}")
    #     sys.exit(1)

    # validate files
    if not validate_model_files(
        converted_files["annotated_entities"],
        converted_files["knowledge_graph"],
        converted_files["entity_embeddings"],
        converted_files["relation_embeddings"],
    ):
        print("❌ Model file validation failed")
        sys.exit(1)

    # set output directory - create results in the project root directory
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Start drug discovery analysis workflow")
    print("=" * 60)

    try:
        # step 1: run full analysis pipeline
        print("\n1. Run full analysis pipeline...")
        core.run_full_pipeline(
            disease_id=disease_id,
            entity_file=converted_files["annotated_entities"],
            knowledge_graph=converted_files["knowledge_graph"],
            entity_embeddings=converted_files["entity_embeddings"],
            relation_embeddings=converted_files["relation_embeddings"],
            output_dir=output_dir,
            model=model_name,
            top_n_diseases=50,
            gamma=gamma,
            threshold=0.5,
            relation_type="GNBR::T::Compound:Disease",
            top_n_drugs=100,
        )

        annotated_file = os.path.join(output_dir, "annotated_drugs.xlsx")
        if os.path.exists(annotated_file):
            print(f"✅ Full analysis completed, results saved to: {annotated_file}")
        else:
            print("❌ Full analysis failed")
            return

        # step 2: apply complex filter expression
        print("\n2. Apply filter conditions...")
        # example filter expression: select drugs with score > 0.6 and num_of_shared_diseases > 2
        filter_expression = "pvalue < 0.05 and num_of_shared_genes_in_path >= 1 and existing == False and num_of_key_genes >= 1 and num_of_shared_pathways >= 1"

        filtered_file = os.path.join(output_dir, "filtered_drugs.xlsx")
        drug_filter.filter_drugs(
            input_file=annotated_file,
            expression=filter_expression,
            output_file=filtered_file,
            sheet_names=("annotated_drugs", "filtered_drugs"),
        )

        if os.path.exists(filtered_file):
            print(f"✅ Filter completed, results saved to: {filtered_file}")
        else:
            print("❌ Filter failed")
            return

        # step 3: generate visualization report
        print("\n3. Generate visualization report...")
        report_dir = os.path.join(output_dir, "visualization_report")
        os.makedirs(report_dir, exist_ok=True)

        # generate comprehensive report
        report_file = os.path.join(report_dir, "analysis_report.html")
        disease_name = core.get_disease_name(disease_id, converted_files["annotated_entities"])
        try:
            visualizer.generate_report(
                data_file=filtered_file,
                output_file=report_file,
                title=f"Drug Discovery Analysis Report - {disease_name}",
            )
            print(f"✅ Generate comprehensive report: {report_file}")
        except Exception as e:
            print(f"❌ Generate report failed: {e}")

        # step 4: explain the drugs
        print("\n4. Explain the drugs...")
        drugs = pd.read_excel(filtered_file, sheet_name="filtered_drugs")
        drug_names = (
            drugs["drug_name"].tolist()[:50]
            if len(drugs) > 50
            else drugs["drug_name"].tolist()
        )
        disease_name = core.get_disease_name(disease_id, converted_files["annotated_entities"])
        explainer = DrugExplain()
        prompt = explainer.generate_prompt(
            drug_names=drug_names, disease_name=disease_name
        )
        print("You can use the following prompt to explain the drugs:")
        print("=" * 60)
        print(prompt)

        print("\n" + "=" * 60)
        print("Analysis completed!")
        print("=" * 60)
        print(f"Main result files:")
        print(f"  - Full analysis results: {annotated_file}")
        print(f"  - Filtered results: {filtered_file}")
        print(f"  - Visualization report: {report_dir}/")
        print(f"  - Comprehensive report: {report_file}")

    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main(disease_id="MONDO:0004979")
