import os
import logging
from drugs4disease.core import DrugDiseaseCore
from drugs4disease.model import Model


logger = logging.getLogger(__name__)

try:
    logger.info("biominer_tools is available, loading biominer_tools...")
    from biominer_tools import (
        Tool,
        ToolParam,
        FieldCategory,
        tool,
        ToolResult,
        ToolFile,
    )
except Exception as e:
    logger.error(f"Error loading biominer_tools: {e}")
    logger.warning(
        f"⚠️ biominer_tools is not available, so this tool will not be registered in biominer_tools."
    )
else:
    @tool(
        name="drugs4disease",
        description="""Predict drugs for a disease.""",
        category="analysis_tool",
        field_category=FieldCategory.DRUG_REPUREPOSING,
        required_params=[
            ToolParam(
                param_name="disease_id",
                param_type=str,
                param_desc="The disease id. It must be a mondo id. such as MONDO:0005148",
                param_default=None,
            ),
        ],
        optional_params=[
            ToolParam(
                param_name="model_run_id",
                param_type=str,
                param_desc="The model run id. It must be a model run id. such as 6vlvgvfq",
                param_default="6vlvgvfq",
            ),
            ToolParam(
                param_name="output_dir",
                param_type=str,
                param_desc="The output directory. It must be a directory. such as ./results",
                param_default=".",
            ),
            ToolParam(
                param_name="top_n_diseases",
                param_type=int,
                param_desc="The number of similar diseases to retrieve. such as 100",
                param_default=100,
            ),
            ToolParam(
                param_name="threshold",
                param_type=float,
                param_desc="The threshold for the drug prediction. such as 0.5",
                param_default=0.5,
            ),
            ToolParam(
                param_name="relation_type",
                param_type=str,
                param_desc="The relation type. It must be a relation type. such as GNBR::T::Compound:Disease",
                param_default="GNBR::T::Compound:Disease",
            ),
            ToolParam(
                param_name="top_n_drugs",
                param_type=int,
                param_desc="The number of drugs to retrieve. such as 100",
                param_default=100,
            ),
        ],
        supported_fields=set(),
        tags={"drugs4disease", "drug", "disease", "drug_repurposing"},
        examples=[
            """drugs4disease(disease_id='MONDO:0005148', model_run_id='6vlvgvfq', output_dir='.', top_n_diseases=100, threshold=0.5, relation_type='GNBR::T::Compound:Disease', top_n_drugs=100)""",
        ],
        return_type="tool_result",
    )
    class DrugDiseaseTool(Tool):
        """Predict drugs for a disease."""

        def execute(self, **kwargs):
            """Execute the tool."""
            output_dir = kwargs.get("output_dir", ".")
            model_run_id = kwargs.get("model_run_id", "6vlvgvfq")
            disease_id = kwargs.get("disease_id", None)
            top_n_diseases = kwargs.get("top_n_diseases", 100)
            threshold = kwargs.get("threshold", 0.5)
            relation_type = kwargs.get("relation_type", "GNBR::T::Compound:Disease")
            top_n_drugs = kwargs.get("top_n_drugs", 100)

            if disease_id is None:
                raise ValueError("disease_id is required")

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
                    top_n_drugs=top_n_drugs,
                )
                logger.info(
                    f'✅ Analysis completed! Results saved to {os.path.join(output_dir, "annotated_drugs.xlsx")}'
                )

                # Check if files exist before creating ToolFile objects
                files = []
                expected_files = [
                    ("predicted_drugs.xlsx", "A spreadsheet of predicted drugs"),
                    ("shared_genes_pathways.xlsx", "A spreadsheet of shared genes and pathways"),
                    ("shared_diseases.xlsx", "A spreadsheet of shared diseases"),
                    ("network_annotations.xlsx", "A spreadsheet of network annotations"),
                    ("annotated_drugs.xlsx", "A spreadsheet of annotated drugs"),
                ]

                for filename, description in expected_files:
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath):
                        files.append(
                            ToolFile(
                                name=os.path.basename(filepath),
                                description=description,
                                filepath=filepath,
                                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                size=os.path.getsize(filepath),
                            )
                        )

                return ToolResult(
                    success=True,
                    files=files,
                    summary=f"✅ Analysis completed! Results saved to {os.path.join(output_dir, 'annotated_drugs.xlsx')}",
                    data=None,
                    error=None,
                    metadata={},
                )
            except Exception as e:
                logger.error(f"Analysis failed: {str(e)}")
                return ToolResult(
                    success=False,
                    files=[],
                    summary="",
                    data=None,
                    error=f"❌ Analysis failed: {str(e)}",
                    metadata={},
                )
