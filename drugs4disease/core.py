import pandas as pd
import numpy as np
import os
from . import lib
import networkx as nx
from gseapy import enrichr
from .utils import get_model_file_paths, validate_model_files
from tqdm import tqdm

class DrugDiseaseCore:
    """
    Responsible for generating potential drug lists and main data processing logic.
    """
    def __init__(self):
        self.logger = lib.get_logger("DrugDiseaseCore")

    def run_full_pipeline(self, disease_id, entity_file=None, knowledge_graph=None, entity_embeddings=None, 
                          relation_embeddings=None, output_dir=None, model='TransE_l2', top_n_diseases=100, gamma=12.0, 
                          threshold=0.5, relation_type='GNBR::T::Compound:Disease', top_n_drugs=1000):
        """
        One-click complete all analysis steps, output annotated_drugs.xlsx, the fields are consistent with the final table in run.ipynb.
        
        Support smart file path processing:
        - If no model file is specified, use the file in the default directory
        - If all four model files are specified, use the specified files
        - Automatically handle ZIP file decompression
        """
        # 智能获取模型文件路径
        entity_file, knowledge_graph, entity_embeddings, relation_embeddings = get_model_file_paths(
            entity_file, knowledge_graph, entity_embeddings, relation_embeddings
        )

        # validate files
        if not validate_model_files(entity_file, knowledge_graph, entity_embeddings, relation_embeddings):
            raise FileNotFoundError("模型文件验证失败")

        # 1. drug prediction
        pred_xlsx = os.path.join(output_dir, 'predicted_drugs.xlsx')
        if os.path.exists(pred_xlsx):
            self.logger.info("Predicted drugs file is found, skip the prediction")
        else:
            self.logger.info("Predicted drugs file is not found, predict the drugs")
            self.predict_drugs(
                disease_id, entity_file, knowledge_graph, entity_embeddings, relation_embeddings,
                model, top_n_diseases, gamma, threshold, relation_type, pred_xlsx
            )

        # 2. shared genes/pathways annotation
        shared_genes_xlsx = os.path.join(output_dir, 'shared_genes_pathways.xlsx')
        if os.path.exists(shared_genes_xlsx):
            self.logger.info("Shared genes/pathways file is found, skip the annotation")
        else:
            self.logger.info("Shared genes/pathways file is not found, annotate the shared genes/pathways")
            self.annotate_shared_genes_pathways(
                pred_xlsx, disease_id, knowledge_graph, shared_genes_xlsx, top_n_drugs
            )

        # 3. shared diseases annotation
        shared_diseases_xlsx = os.path.join(output_dir, 'shared_diseases.xlsx')
        if os.path.exists(shared_diseases_xlsx):
            self.logger.info("Shared diseases file is found, skip the annotation")
        else:
            self.logger.info("Shared diseases file is not found, annotate the shared diseases")
            self.annotate_shared_diseases(
                pred_xlsx, disease_id, knowledge_graph, entity_embeddings, relation_embeddings, shared_diseases_xlsx, model, gamma, top_n_diseases
            )

        # 4. network analysis/centrality/pathway annotation
        network_anno_xlsx = os.path.join(output_dir, 'network_annotations.xlsx')
        if os.path.exists(network_anno_xlsx):
            self.logger.info("Network annotations file is found, skip the annotation")
        else:
            self.logger.info("Network annotations file is not found, annotate the network annotations")
            self.annotate_network_features(
                pred_xlsx, disease_id, knowledge_graph, entity_file, network_anno_xlsx, top_n_drugs
            )

        # 5. merge all annotations, output annotated_drugs.xlsx
        annotated_xlsx = os.path.join(output_dir, 'annotated_drugs.xlsx')
        if os.path.exists(annotated_xlsx):
            self.logger.info("Annotated drugs file is found, skip the merge")
        else:
            self.logger.info("Annotated drugs file is not found, merge the annotations")
            self.merge_annotations(
                pred_xlsx, shared_genes_xlsx, shared_diseases_xlsx, network_anno_xlsx, annotated_xlsx
            )
        self.logger.info(f'Generated {annotated_xlsx}')

    @staticmethod
    def get_disease_name(disease_id: str, entity_file: str) -> str:
        """
        Get the disease name.
        """
        entity_df = pd.read_csv(entity_file, sep="\t", dtype=str)
        disease_name = entity_df[entity_df["id"] == disease_id]["name"].to_numpy()[0]
        return disease_name

    @staticmethod
    def get_drug_names(drug_ids: list[str], entity_file: str) -> list[str]:
        """
        Get the drug names.
        """
        entity_df = pd.read_csv(entity_file, sep="\t", dtype=str)
        drug_names = entity_df[entity_df["id"].isin(drug_ids)]["name"].to_numpy()
        return drug_names.tolist()

    def predict_drugs(self, disease_id, entity_file, knowledge_graph, entity_embeddings, relation_embeddings, model, top_n_diseases, gamma, threshold, relation_type, output_file):
        """
        Generate potential drug list, save annotated_drugs.xlsx.
        """
        if not output_file.endswith(".xlsx"):
            raise ValueError("Output file must be an Excel file")

        # Load the knowledge graph
        kg_df = pd.read_csv(knowledge_graph, sep="\t", dtype=str)
        if not set([
            "source_id", "source_type", "target_id", "target_type", "relation_type"
        ]).issubset(set(kg_df.columns)):
            raise ValueError(
                "The knowledge graph file must have the following columns: source_id, source_type, source_name, target_id, target_type, target_name, relation_type"
            )
        self.logger.info("Knowledge graph file is loaded")

        # BioMedGPS format knowledge graph already contains source_name and target_name
        if not set(["source_name", "target_name"]).issubset(set(kg_df.columns)):
            # if the name column is missing, load from annotated_entities.tsv
            entity_df = pd.read_csv(entity_file, sep="\t")
            entity_df = entity_df[["id", "label", "name"]]
            entity_df_source = entity_df.rename(columns={"id": "source_id", "label": "source_type", "name": "source_name"})
            entity_df_target = entity_df.rename(columns={"id": "target_id", "label": "target_type", "name": "target_name"})
            kg_df = kg_df.merge(entity_df_source, on=["source_id", "source_type"], how="left")
            kg_df = kg_df.merge(entity_df_target, on=["target_id", "target_type"], how="left")

        drug_df = kg_df[(kg_df["source_type"] == "Compound") | (kg_df["target_type"] == "Compound")]
        source_drugs = drug_df[["source_id", "source_name"]].rename(columns={"source_id": "drug_id", "source_name": "drug_name"})
        target_drugs = drug_df[["target_id", "target_name"]].rename(columns={"target_id": "drug_id", "target_name": "drug_name"})
        drugs = pd.concat([source_drugs, target_drugs])
        drugs.drop_duplicates(inplace=True)
        self.logger.info(f"Drugs extracted: {len(drugs)} rows")

        # Load entity embeddings (BioMedGPS format)
        entity_embeddings_df = pd.read_csv(entity_embeddings, sep="\t")
        # BioMedGPS format embedding column is already a string format vector
        entity_embeddings_df["embedding"] = entity_embeddings_df["embedding"].apply(lambda x: np.array(x.split("|"), dtype=float))
        self.logger.info("Entity embeddings file is loaded")

        # Load relation embeddings (BioMedGPS format)
        relation_embeddings_df = pd.read_csv(relation_embeddings, sep="\t")
        # BioMedGPS format embedding column is already a string format vector
        relation_embeddings_df["embedding"] = relation_embeddings_df["embedding"].apply(lambda x: np.array(x.split("|"), dtype=float))
        self.logger.info("Relation embeddings file is loaded")

        if disease_id not in entity_embeddings_df[entity_embeddings_df["entity_type"] == "Disease"]["entity_id"].to_numpy():
            raise ValueError("The disease id is not in the entity file.")

        target_disease_embedding = entity_embeddings_df[(entity_embeddings_df["entity_id"] == disease_id) & (entity_embeddings_df["entity_type"] == "Disease")]["embedding"].to_numpy()[0]

        # BioMedGPS format relation ID lookup
        relation_embedding = None
        if "id" in relation_embeddings_df.columns:
            relation_embedding = relation_embeddings_df[relation_embeddings_df["id"] == relation_type]["embedding"].to_numpy()[0]
        elif "relation_type" in relation_embeddings_df.columns:
            relation_embedding = relation_embeddings_df[relation_embeddings_df["relation_type"] == relation_type]["embedding"].to_numpy()[0]
        else:
            # if the relation embedding is not found, use the default vector
            self.logger.warning(f"Relation type {relation_type} not found in embeddings, using default")
            relation_embedding = np.zeros_like(target_disease_embedding)

        all_drugs = entity_embeddings_df[entity_embeddings_df["entity_type"] == "Compound"]["entity_id"].to_numpy()
        all_drugs = list(set(all_drugs))
        all_drugs = set(all_drugs).intersection(set(drugs["drug_id"]))

        self.logger.info("Compute the scores for all drugs")
        all_drugs_df = drugs[drugs["drug_id"].isin(all_drugs)]
        all_drugs_df.reset_index(drop=True, inplace=True)
        drug_embeddings = entity_embeddings_df[entity_embeddings_df["entity_type"] == "Compound"][["entity_id", "embedding"]]
        all_drugs_df = all_drugs_df[["drug_id", "drug_name"]].merge(drug_embeddings, how="left", left_on="drug_id", right_on="entity_id")
        all_drugs_df.drop(columns=["entity_id"], inplace=True)

        drug_names = all_drugs_df["drug_name"].to_numpy()
        all_drugs = all_drugs_df["drug_id"].to_numpy()
        head_embeddings = list(all_drugs_df["embedding"].to_numpy())

        scores = pd.DataFrame({
            "drug_id": all_drugs,
            "drug_name": drug_names,
            "score": lib.kge_score_fn_batch(
                head_embeddings,
                relation_embedding,
                [target_disease_embedding],
                gamma=gamma,
                model=model,
            ),
        })

        self.logger.info("Predicted drugs are sorted by the score")
        scores.sort_values(by=["score"], ascending=False, inplace=True)
        scores.reset_index(drop=True, inplace=True)
        scores["rank"] = scores.index + 1
        scores["pvalue"] = scores["rank"] / len(scores)
        scores["pvalue"] = scores["pvalue"].apply(lambda x: f"{x:.3g}")
        lib.save_df(scores, output_file, "predicted_drugs")

        self.logger.info("Get the top N drugs for the given disease.")
        filtered_scores = scores[scores["score"] > threshold][["drug_id", "drug_name"]]
        if filtered_scores.empty:
            filtered_scores = scores.head(100)[["drug_id", "drug_name"]]
        suffix = relation_type.split("::")[0].lower()
        drug_id_file = os.path.join(os.path.dirname(output_file), f"top_n_drugs_{suffix}.tsv")
        filtered_scores.to_csv(drug_id_file, sep="\t", index=False)

        self.logger.info("Get the top N diseases for the given disease")
        disease_embedding = entity_embeddings_df[entity_embeddings_df["entity_id"] == disease_id]["embedding"].to_numpy()[0]
        all_disease_embeddings = list(entity_embeddings_df[entity_embeddings_df["entity_type"] == "Disease"]["embedding"].to_numpy())
        all_disease_ids = entity_embeddings_df[entity_embeddings_df["entity_type"] == "Disease"]["entity_id"].to_numpy()

        self.logger.info(f"The type of disease_embedding: {type(disease_embedding)}, len(disease_embedding): {len(disease_embedding)}")
        self.logger.info(f"The type of all_disease_embeddings: {type(all_disease_embeddings)}, len(all_disease_embeddings): {len(all_disease_embeddings)}")
        self.logger.info(f"The type of relation_embedding: {type(relation_embedding)}, len(relation_embedding): {len(relation_embedding)}")

        (
            top_n_similar_diseases_indices,
            top_n_similar_diseases_scores,
        ) = lib.compute_top_n_similar_entities_vectorized(
            disease_embedding,
            all_disease_embeddings,
            relation_embedding,
            top_n=top_n_diseases,
            model=model,
            gamma=gamma,
        )
        top_n_disease_ids = all_disease_ids[top_n_similar_diseases_indices]
        top_n_disease_ids = top_n_disease_ids[top_n_disease_ids != disease_id]

        self.logger.info("Get the treated diseases for the predicted drugs")
        treated_disease_df = kg_df[
            (kg_df["source_id"].isin(all_drugs))
            & (kg_df["relation_type"].isin([relation_type]))
        ]
        treated_disease_df = treated_disease_df[["source_id", "source_name", "target_id", "target_name"]]
        treated_disease_df.rename(
            columns={
                "source_id": "drug_id",
                "source_name": "drug_name",
                "target_id": "disease_id",
                "target_name": "disease_name",
            },
            inplace=True,
        )
        treated_disease_df["drug_id_count"] = treated_disease_df["drug_id"]
        treated_disease_grouped_df = treated_disease_df.groupby(["drug_id", "drug_name"], as_index=False).agg({
            "disease_id": lambda x: list(x),
            "disease_name": lambda x: list(x),
            "drug_id_count": "count",
        })
        treated_disease_grouped_df.rename(
            columns={
                "drug_id": "drug_id",
                "drug_name": "drug_name",
                "disease_id": "treated_disease_ids",
                "disease_name": "treated_disease_names",
                "drug_id_count": "num_of_treated_diseases",
            },
            inplace=True,
        )
        treated_disease_grouped_df["treated_similar_disease_ids"] = treated_disease_grouped_df["treated_disease_ids"].apply(lambda x: "|".join(list(set(x).intersection(set(top_n_disease_ids)))))
        treated_disease_grouped_df["treated_similar_disease_names"] = treated_disease_grouped_df["treated_similar_disease_ids"].apply(lambda x: treated_disease_df[treated_disease_df["disease_id"].isin(x.split("|"))]["disease_name"].to_numpy())
        treated_disease_grouped_df["treated_similar_disease_names"] = treated_disease_grouped_df["treated_similar_disease_names"].apply(lambda x: "|".join(list(set(x))))
        treated_disease_grouped_df["num_of_treated_similar_diseases"] = treated_disease_grouped_df["treated_similar_disease_ids"].apply(lambda x: len([i for i in x.split("|") if i]))
        treated_disease_grouped_df["treated_disease_ids"] = treated_disease_grouped_df["treated_disease_ids"].apply(lambda x: "|".join(x))
        treated_disease_grouped_df["treated_disease_names"] = treated_disease_grouped_df["treated_disease_names"].apply(lambda x: "|".join(x))
        treated_disease_grouped_df.sort_values(by=["num_of_treated_similar_diseases"], ascending=False, inplace=True)
        lib.save_df(treated_disease_grouped_df, output_file, "drug_with_treated_diseases")

    def annotate_shared_genes_pathways(self, predicted_drug_file: str, disease_id: str, knowledge_graph: str, output_file: str, top_n_drugs: int = 1000):
        """
        Count the shared genes and pathways of drugs and diseases, output shared_genes_pathways.xlsx.
        """
        if not output_file.endswith(".xlsx"):
            raise ValueError("Output file must be an Excel file")

        kg_df = pd.read_csv(knowledge_graph, sep="\t", dtype=str)
        if not set([
            "source_id", "source_type", "source_name", "target_id", "target_type", "target_name", "relation_type"
        ]).issubset(set(kg_df.columns)):
            raise ValueError(
                "The knowledge graph file must have the following columns: source_id, source_type, source_name, target_id, target_type, target_name, relation_type"
            )
        self.logger.info("Knowledge graph file is loaded")

        known_diseases = [
            {
                "id": disease_id,
                "name": kg_df[kg_df["source_id"] == disease_id]["source_name"].to_numpy()[0],
            }
        ]
        known_df = pd.DataFrame(known_diseases)

        predicted_df = pd.read_excel(predicted_drug_file, sheet_name="predicted_drugs")
        if not set(["drug_id", "drug_name"]).issubset(set(predicted_df.columns)):
            raise ValueError(
                "The predicted drug file must have the following columns: drug_id, drug_name"
            )
        self.logger.info(f"Predicted drug file loaded: {len(predicted_df)} rows")
        predicted_df.drop_duplicates(subset=["drug_id"], inplace=True)
        predicted_df.drop_duplicates(subset=["drug_name"], inplace=True)
        predicted_df.reset_index(drop=True, inplace=True)

        drug_gene_relations = kg_df[
            ((kg_df["source_type"] == "Compound") & (kg_df["target_type"] == "Gene")) |
            ((kg_df["source_type"] == "Gene") & (kg_df["target_type"] == "Compound"))
        ]
        disease_gene_relations = kg_df[
            (((kg_df["source_type"] == "Disease") & (kg_df["source_id"] == disease_id) & (kg_df["target_type"] == "Gene")) |
             ((kg_df["target_type"] == "Disease") & (kg_df["target_id"] == disease_id) & (kg_df["source_type"] == "Gene")))
        ]
        predicted_drug_gene_relations = drug_gene_relations[
            drug_gene_relations["source_id"].isin(predicted_df["drug_id"]) |
            drug_gene_relations["target_id"].isin(predicted_df["drug_id"])
        ]

        def get_shared_genes():
            return [
                set(lib.get_genes(
                    disease_gene_relations[
                        (disease_gene_relations["source_id"] == x) |
                        (disease_gene_relations["target_id"] == x)
                    ]
                )).intersection(
                    lib.get_genes(
                        predicted_drug_gene_relations[
                            (predicted_drug_gene_relations["source_id"] == y) |
                            (predicted_drug_gene_relations["target_id"] == y)
                        ]
                    )
                )
                for x in known_df["id"]
                for y in predicted_df["drug_id"]
            ]

        def get_shared_gene_names():
            return [
                set(lib.get_gene_names(
                    disease_gene_relations[
                        (disease_gene_relations["source_id"] == x) |
                        (disease_gene_relations["target_id"] == x)
                    ]
                )).intersection(
                    lib.get_gene_names(
                        predicted_drug_gene_relations[
                            (predicted_drug_gene_relations["source_id"] == y) |
                            (predicted_drug_gene_relations["target_id"] == y)
                        ]
                    )
                )
                for x in known_df["id"]
                for y in predicted_df["drug_id"]
            ]

        self.logger.info("Get the shared genes")

        if os.path.exists(output_file):
            self.logger.info("Shared genes dataframe is loaded")
            shared_genes_df = pd.read_excel(output_file, sheet_name="shared_genes")
        else:
            # pre-compute shared genes to avoid duplicate calculation
            shared_genes_list = get_shared_genes()
            shared_gene_names_list = get_shared_gene_names()

            shared_genes_df = pd.DataFrame({
                "known_disease_id": np.repeat(known_df["id"], len(predicted_df["drug_id"])),
                "known_disease_name": np.repeat(known_df["name"], len(predicted_df["drug_name"])),
                "predicted_drug_id": predicted_df["drug_id"].tolist() * len(known_df["id"]),
                "predicted_drug_name": predicted_df["drug_name"].tolist() * len(known_df["name"]),
                "shared_genes": ["|".join(x) for x in shared_genes_list],
                "shared_gene_names": ["|".join(x) for x in shared_gene_names_list],
                "num_of_shared_genes": [len(x) for x in shared_genes_list],
            })

            self.logger.info("Shared genes dataframe is created")
            lib.save_df(shared_genes_df, output_file, "shared_genes")

        known_disease_enrichment_results = []   
        for i, row in tqdm(known_df.iterrows(), total=len(known_df), desc="Performing pathway enrichment analysis for known diseases"):
            genes = lib.get_genes(
                disease_gene_relations[
                    (disease_gene_relations["source_id"] == known_df["id"][i]) |
                    (disease_gene_relations["target_id"] == known_df["id"][i])
                ]
            )
            if len(genes) == 0:
                continue
            genes = list(map(lambda x: x.replace("ENTREZ:", ""), genes))
            result = lib.pathway_enrichment(genes)
            result["known_disease_id"] = known_df["id"][i]
            result["known_disease_name"] = known_df["name"][i]
            known_disease_enrichment_results.append(result)
        if known_disease_enrichment_results:
            known_disease_enrichment_results = pd.concat(known_disease_enrichment_results)
            lib.save_df(known_disease_enrichment_results, output_file, "target_disease_pathways")
        else:
            known_disease_enrichment_results = pd.DataFrame()

        predicted_drug_enrichment_results = []
        for i, row in tqdm(predicted_df[:top_n_drugs].iterrows(), total=top_n_drugs, desc="Performing pathway enrichment analysis for predicted drugs"):
            genes = lib.get_genes(
                predicted_drug_gene_relations[
                    (predicted_drug_gene_relations["source_id"] == predicted_df["drug_id"][i]) |
                    (predicted_drug_gene_relations["target_id"] == predicted_df["drug_id"][i])
                ]
            )
            if len(genes) == 0:
                continue
            genes = list(map(lambda x: x.replace("ENTREZ:", ""), genes))
            result = lib.pathway_enrichment(genes)
            result["drug_id_predicted"] = predicted_df["drug_id"][i]
            result["drug_name_predicted"] = predicted_df["drug_name"][i]
            predicted_drug_enrichment_results.append(result)

        if predicted_drug_enrichment_results:
            predicted_drug_enrichment_results = pd.concat(predicted_drug_enrichment_results)
            lib.save_df(predicted_drug_enrichment_results, output_file, "predicted_drug_pathways")
        else:
            predicted_drug_enrichment_results = pd.DataFrame()

        self.logger.info("Pathway enrichment analysis is done")
        if not known_disease_enrichment_results.empty and not predicted_drug_enrichment_results.empty:
            shared_pathways = pd.merge(
                known_disease_enrichment_results,
                predicted_drug_enrichment_results,
                on=["source", "native", "name", "description"],
                suffixes=("_known", "_predicted"),
            )
            shared_pathways.rename(
                columns={
                    "source": "pathway_source",
                    "native": "pathway_id",
                    "name": "pathway_name",
                    "description": "pathway_description",
                    "known_disease_id": "known_disease_id",
                    "known_disease_name": "known_disease_name",
                    "drug_id_predicted": "predicted_drug_id",
                    "drug_name_predicted": "predicted_drug_name",
                },
                inplace=True,
            )
            shared_pathways = shared_pathways[
                [
                    "known_disease_id",
                    "known_disease_name",
                    "predicted_drug_id",
                    "predicted_drug_name",
                    "pathway_id",
                    "pathway_name",
                    "pathway_description",
                    "pathway_source",
                ]
            ]
            lib.save_df(shared_pathways, output_file, "shared_pathways")
            shared_pathways["pathway_id_count"] = shared_pathways["pathway_id"]
            shared_pathways = shared_pathways[
                (shared_pathways["pathway_source"] == "KEGG") |
                (shared_pathways["pathway_source"] == "REAC") |
                (shared_pathways["pathway_source"] == "WP")
            ]
            shared_pathways = (
                shared_pathways.groupby(
                    [
                        "known_disease_id",
                        "known_disease_name",
                        "predicted_drug_id",
                        "predicted_drug_name",
                    ]
                )
                .agg(
                    {
                        "pathway_id_count": "count",
                        "pathway_id": lambda x: "|".join(list(set(x))),
                        "pathway_name": lambda x: "|".join(list(set(x))),
                        "pathway_description": lambda x: "|".join(list(set(x))),
                        "pathway_source": lambda x: "|".join(list(set(x))),
                    }
                )
                .reset_index()
            )
            shared_pathways.rename(
                columns={"pathway_id_count": "number_of_shared_pathways"}, inplace=True
            )
            lib.save_df(shared_pathways, output_file, "shared_pathways_summary")

    def annotate_shared_diseases(self, predicted_drug_file, disease_id, knowledge_graph, entity_embeddings, relation_embeddings, output_file, model, gamma, top_n):
        """
        Count the shared diseases of drugs and diseases, output shared_diseases.xlsx.
        """
        similar_disease_relation_types = ["BioMedGPS::SimilarWith::Disease:Disease", "Hetionet::DrD::Disease:Disease"]
        treated_disease_relation_types = ["BioMedGPS::Treatment::Compound:Disease", "GNBR::T::Compound:Disease", "DrugBank::treats::Compound:Disease"]

        if not output_file.endswith(".xlsx"):
            raise ValueError("Output file must be an Excel file")

        entities_df = pd.read_csv(entity_embeddings, sep="\t")
        if not set(["entity_id", "entity_type", "embedding"]).issubset(set(entities_df.columns)):
            raise ValueError("The entity embeddings file must have the following columns: entity_id, entity_type, embedding")

        entities_df["embedding"] = entities_df["embedding"].apply(lambda x: np.array(x.split("|"), dtype=float))
        self.logger.info("Entity embeddings file is loaded")

        diseases = entities_df[entities_df["entity_type"] == "Disease"]["entity_id"].to_numpy().tolist()
        if disease_id not in diseases:
            raise Exception("The disease id is not in the entity file.")

        relations_df = pd.read_csv(relation_embeddings, sep="\t")
        if not set(["id", "embedding"]).issubset(set(relations_df.columns)):
            raise ValueError("The relation embeddings file must have the following columns: id, embedding")

        relations_df["embedding"] = relations_df["embedding"].apply(lambda x: np.array(x.split("|"), dtype=float))
        self.logger.info("Relation embeddings file is loaded")

        relation_embedding = None
        for similar_disease_relation_type in similar_disease_relation_types:
            try:
                relation_embedding = relations_df[relations_df["id"] == similar_disease_relation_type]["embedding"].to_numpy()[0]
                if relation_embedding is not None and relation_embedding.shape[0] > 0:
                    break
            except Exception as e:
                self.logger.warning(f"The relation type {similar_disease_relation_type} is not found in the relation embeddings file")
                continue

        predicted_df = pd.read_excel(predicted_drug_file, sheet_name="predicted_drugs")
        if not set(["drug_id", "drug_name"]).issubset(set(predicted_df.columns)):
            raise ValueError("The predicted file must have the following columns: drug_id, drug_name")

        self.logger.info(f"Predicted file is loaded, and it has {len(predicted_df)} rows")
        predicted_df.drop_duplicates(subset=["drug_id"], inplace=True)
        predicted_df.drop_duplicates(subset=["drug_name"], inplace=True)
        self.logger.info(f"Predicted file is deduplicated, and it has {len(predicted_df)} rows")

        kg_df = pd.read_csv(knowledge_graph, sep="\t")
        if not set(["source_id", "source_type", "source_name", "target_id", "target_type", "target_name", "relation_type"]).issubset(set(kg_df.columns)):
            raise ValueError("The knowledge graph file must have the following columns: source_id, source_type, source_name, target_id, target_type, target_name, relation_type")
        self.logger.info("Knowledge graph file is loaded")

        self.logger.info("Get the top N similar diseases for the given disease")
        disease_embedding = entities_df[entities_df["entity_id"] == disease_id]["embedding"].to_numpy()[0]
        all_disease_embeddings = list(entities_df[entities_df["entity_type"] == "Disease"]["embedding"].to_numpy())
        all_disease_ids = entities_df[entities_df["entity_type"] == "Disease"]["entity_id"].to_numpy()

        self.logger.info(f"The type of disease_embedding: {type(disease_embedding)}, len(disease_embedding): {len(disease_embedding)}")
        self.logger.info(f"The type of all_disease_embeddings: {type(all_disease_embeddings)}, len(all_disease_embeddings): {len(all_disease_embeddings)}")
        self.logger.info(f"The type of relation_embedding: {type(relation_embedding)}, len(relation_embedding): {len(relation_embedding)}")

        (top_n_similar_diseases_indices, top_n_similar_diseases_scores) = lib.compute_top_n_similar_entities_vectorized(
            disease_embedding, all_disease_embeddings, relation_embedding, top_n=top_n, model=model, gamma=gamma
        )
        top_n_disease_ids = all_disease_ids[top_n_similar_diseases_indices]

        similar_diseases_df = pd.DataFrame(columns=["similar_disease_id", "disease_id", "score"])
        similar_diseases_df["similar_disease_id"] = top_n_disease_ids
        similar_diseases_df["disease_id"] = disease_id
        similar_diseases_df["score"] = top_n_similar_diseases_scores

        similar_diseases_df = pd.merge(
            similar_diseases_df,
            entities_df[["entity_id", "entity_name"]],
            left_on=["similar_disease_id"],
            right_on=["entity_id"],
            how="left",
        )
        similar_diseases_df.rename(
            columns={"entity_id": "similar_disease_id", "entity_name": "similar_disease_name"}, inplace=True
        )

        lib.save_df(similar_diseases_df, output_file, "similar_diseases")
        self.logger.info("Similar diseases are saved to the file")

        predicted_drug_ids = predicted_df["drug_id"].to_numpy()
        predicted_drug_names = predicted_df["drug_name"].to_numpy()

        self.logger.info("Get the treated diseases for the predicted drugs")
        results = []
        # pre-filter the related knowledge graph data to avoid duplicate filtering in the loop
        treated_disease_df = kg_df[
            (kg_df["source_id"].isin(predicted_drug_ids)) & 
            (kg_df["relation_type"].isin(treated_disease_relation_types))
        ][["source_id", "target_id", "target_name"]]

        # group by drug
        grouped = treated_disease_df.groupby("source_id")

        # use vectorized operation to process each drug
        for index, drug in tqdm(enumerate(predicted_drug_ids), total=len(predicted_drug_ids), desc="Get the treated diseases for the predicted drugs"):
            if drug in grouped.groups:
                drug_data = grouped.get_group(drug)
                treated_disease_ids = drug_data["target_id"].to_numpy()

                # use vectorized intersection operation
                shared_disease_ids = np.intersect1d(treated_disease_ids, top_n_disease_ids)

                if len(shared_disease_ids) > 0:
                    # only query the shared disease names, avoid querying all diseases
                    shared_disease_names = drug_data[drug_data["target_id"].isin(shared_disease_ids)]["target_name"].to_numpy()
                    shared_names_str = "|".join(list(set(shared_disease_names)))
                else:
                    shared_names_str = ""
            else:
                treated_disease_ids = np.array([])
                shared_disease_ids = np.array([])
                shared_names_str = ""

            results.append({
                "drug_id": drug,
                "drug_name": predicted_drug_names[index],
                "num_of_treated_diseases": len(treated_disease_ids),
                "num_of_shared_diseases": len(shared_disease_ids),
                "shared_disease_names": shared_names_str,
            })

        self.logger.info("Stat the shared diseases")
        results_df = pd.DataFrame(results)
        lib.save_df(results_df, output_file, "shared_diseases")
        self.logger.info("The results are saved to the file")

    def annotate_network_features(self, predicted_drug_file, disease_id, knowledge_graph, entity_file, output_file, top_n_drugs=1000):
        """
        Calculate the network annotations of drug-disease-gene path, centrality, PPI, pathway enrichment, etc., output network_annotations.xlsx.
        """
        if not output_file.endswith(".xlsx"):
            raise ValueError("Output file must be an Excel file")

        # read knowledge graph
        kg_df = pd.read_csv(knowledge_graph, sep="\t", dtype=str)
        kg_df["source"] = kg_df["source_type"] + "::" + kg_df["source_id"]
        kg_df["target"] = kg_df["target_type"] + "::" + kg_df["target_id"]

        # read entity file
        entities_df = pd.read_csv(entity_file, sep="\t", dtype=str)
        entities_df["id"] = entities_df["label"] + "::" + entities_df["id"]
        entities_df["xrefs"] = entities_df["xrefs"].astype(str)
        entities_df["symbol"] = entities_df["xrefs"].apply(
            lambda x: next((item.split(":")[1] for item in x.split("|") if item.startswith("SYMBOL:")), None)
        )
        entities_df["symbol"] = entities_df["symbol"].fillna(entities_df["name"])
        id_to_name = dict(zip(entities_df["id"], entities_df["symbol"]))

        # read predicted drugs
        drugs_df = pd.read_excel(predicted_drug_file, sheet_name="predicted_drugs")
        drugs_df["id"] = "Compound::" + drugs_df["drug_id"]
        all_drugs = drugs_df["id"].tolist()

        # target disease
        target_disease = f"Disease::{disease_id}"
        allowed_types = {"Disease", "Gene", "Pathway"}

        # create directed graph
        self.logger.info("Create the directed graph...")
        G = nx.MultiDiGraph()
        for _, row in kg_df.iterrows():
            G.add_edge(row["source"], row["target"], relation=row["relation_type"])

        self.logger.info("Store the drug to disease all two-hop paths...")
        # store all two-hop paths of drug to disease
        drug_disease_paths = {}
        for drug in all_drugs:
            paths = []
            if drug in G:
                for node1 in G.successors(drug):
                    if not any(node1.startswith(t + "::") for t in allowed_types):
                        continue
                    for node2 in G.successors(node1):
                        if node2 == target_disease:
                            paths.append([drug, node1, node2])
            if paths:
                drug_disease_paths[drug] = paths

        self.logger.info("Parse the path data...")
        # parse path data
        formatted_results = []
        for drug, paths in drug_disease_paths.items():
            drug_name = id_to_name.get(drug, drug)
            for path in paths:
                path_names = [id_to_name.get(node_id, node_id) for node_id in path]
                formatted_results.append([drug.split("::")[1], drug_name, path_names[1], path_names[2], disease_id])

        df_results = pd.DataFrame(formatted_results, columns=["drug_id", "drug_name", "path", "disease_name", "disease_id"])

        self.logger.info("Stat the path count...")
        # count the path number
        df = df_results.groupby(["drug_id", "disease_id"]).size().reset_index(name="path_count")
        drugs_df["num_of_shared_genes_in_path"] = drugs_df["drug_id"].map(df.set_index("drug_id")["path_count"])
        drugs_df["num_of_shared_genes_in_path"] = drugs_df["num_of_shared_genes_in_path"].fillna(0).astype(int)

        df = df_results.groupby(["drug_id", "disease_id"]).agg({"path": lambda x: ";".join(x)}).reset_index()
        drugs_df["paths"] = drugs_df["drug_id"].map(df.set_index("drug_id")["path"])

        self.logger.info("Check the existing drugs...")
        # check the existing drugs
        existing_drugs = kg_df[
            ((kg_df["target"] == target_disease) & (kg_df["source_type"] == "Compound")) |
            ((kg_df["source"] == target_disease) & (kg_df["target_type"] == "Compound"))
        ]
        existing_drugs = existing_drugs["source_id"].tolist() + existing_drugs["target_id"].tolist()
        existing_drugs = list(set(existing_drugs))
        drugs_df["existing"] = drugs_df["drug_id"].isin(existing_drugs)

        self.logger.info("Perform the pathway enrichment analysis...")
        # pathway enrichment analysis
        specific_disease_gene = kg_df[
            ((kg_df["source_id"] == disease_id) & (kg_df["target_type"] == "Gene")) |
            ((kg_df["target_id"] == disease_id) & (kg_df["source_type"] == "Gene"))
        ]
        specific_disease_genes = set(
            specific_disease_gene["target_id"][specific_disease_gene["source_id"] == disease_id]
        ).union(set(specific_disease_gene["source_id"][specific_disease_gene["target_id"] == disease_id]))

        self.logger.info("Extract the drug-gene edges...")
        # extract the drug-gene edges
        drug_gene_edges = kg_df[
            ((kg_df["source_type"] == "Compound") & (kg_df["target_type"] == "Gene")) |
            ((kg_df["source_type"] == "Gene") & (kg_df["target_type"] == "Compound"))
        ]

        self.logger.info("Perform the disease pathway enrichment analysis...")
        # disease pathway enrichment
        specific_disease_gene_symbols = [id_to_name.get("Gene::" + gene, gene) for gene in specific_disease_genes]
        self.logger.info(f"The specific disease gene symbols: {specific_disease_gene_symbols[:10]}")
        enrich_results = enrichr(
            gene_list=list(specific_disease_gene_symbols),
            gene_sets=["KEGG_2021_Human"],
            outdir=None,
        )
        enrich_df = enrich_results.results
        enrich_df = enrich_df[enrich_df["Adjusted P-value"] < 0.05]

        self.logger.info("Perform the drug pathway enrichment and overlap analysis...")
        # drug pathway enrichment and overlap analysis
        for index, drug in tqdm(drugs_df.head(top_n_drugs).iterrows(), total=top_n_drugs, desc="Drug pathway enrichment and overlap analysis"):
            drug_id, drug_name = drug["drug_id"], drug["drug_name"]

            # drug-regulated gene list
            regulated_genes = set(
                drug_gene_edges["target_id"][drug_gene_edges["source_id"] == drug_id]
            ).union(set(drug_gene_edges["source_id"][drug_gene_edges["target_id"] == drug_id]))

            regulated_gene_symbols = [id_to_name.get("Gene::" + gene, gene) for gene in regulated_genes]

            if len(regulated_gene_symbols) < 5:
                continue

            # drug pathway enrichment
            drug_enrich_results = enrichr(
                gene_list=list(regulated_gene_symbols),
                gene_sets=["KEGG_2021_Human"],
                outdir=None,
            )
            drug_enrich_df = drug_enrich_results.results
            drug_enrich_df = drug_enrich_df[drug_enrich_df["Adjusted P-value"] < 0.05]

            # calculate the overlap pathways
            shared_pathways = set(drug_enrich_df["Term"]) & set(enrich_df["Term"])
            shared_pathways = list(shared_pathways)
            shared_pathways.sort()
            drugs_df.loc[index, "num_of_shared_pathways"] = len(shared_pathways)
            drugs_df.loc[index, "shared_pathways"] = ";".join(shared_pathways)

        self.logger.info("Perform the PPI network analysis...")
        # PPI network analysis
        gene_gene_edges = kg_df[(kg_df["source_type"] == "Gene") & (kg_df["target_type"] == "Gene")]
        ppi_network = nx.from_pandas_edgelist(gene_gene_edges, "source_id", "target_id")

        for index, drug in tqdm(drugs_df.head(top_n_drugs).iterrows(), total=top_n_drugs, desc="PPI network analysis"):
            drug_id, drug_name = drug["drug_id"], drug["drug_name"]

            # drug-regulated gene
            regulated_genes = set(
                kg_df["target_id"][(kg_df["source_id"] == drug_id) & (kg_df["target_type"] == "Gene")]
            ).union(set(kg_df["source_id"][(kg_df["target_id"] == drug_id) & (kg_df["source_type"] == "Gene")]))

            # extract the drug-regulated gene subgraph
            subgraph_genes = [gene for gene in regulated_genes if gene in ppi_network]
            subgraph = ppi_network.subgraph(subgraph_genes)

            if subgraph.number_of_edges() == 0:
                drugs_df.loc[index, "num_of_key_genes"] = 0
                drugs_df.loc[index, "key_genes"] = ""
                continue

            # calculate the network centrality metrics
            degree_centrality = nx.degree_centrality(subgraph)
            betweenness_centrality = nx.betweenness_centrality(subgraph)

            centrality_df = pd.DataFrame({
                "gene_id": list(degree_centrality.keys()),
                "degree_centrality": list(degree_centrality.values()),
                "betweenness_centrality": list(betweenness_centrality.values()),
            })

            centrality_df.sort_values(["degree_centrality", "betweenness_centrality"], ascending=False, inplace=True)
            centrality_df["gene_name"] = centrality_df["gene_id"].apply(lambda x: id_to_name.get("Gene::" + x, x))

            # only keep the genes related to the disease
            centrality_df = centrality_df[centrality_df["gene_id"].isin(specific_disease_genes)]

            if centrality_df.empty:
                drugs_df.loc[index, "num_of_key_genes"] = 0
                drugs_df.loc[index, "key_genes"] = ""
                continue

            # determine the key genes (top 10%)
            threshold = centrality_df["degree_centrality"].quantile(0.90)
            centrality_df = centrality_df[centrality_df["degree_centrality"] >= threshold]

            threshold = centrality_df["betweenness_centrality"].quantile(0.90)
            centrality_df = centrality_df[centrality_df["betweenness_centrality"] >= threshold]

            num_of_key_genes = len(centrality_df)
            drugs_df.loc[index, "num_of_key_genes"] = num_of_key_genes
            drugs_df.loc[index, "key_genes"] = ";".join(centrality_df["gene_name"].tolist())

        self.logger.info("Perform the drug degree calculation...")
        # calculate the drug degree
        for index, drug in tqdm(drugs_df.head(top_n_drugs).iterrows(), total=top_n_drugs, desc="Drug degree calculation"):
            drug_id = drug["drug_id"]
            drug_degree = kg_df[(kg_df["source_id"] == drug_id) | (kg_df["target_id"] == drug_id)]
            drugs_df.loc[index, "drug_degree"] = drug_degree.shape[0]

        self.logger.info("Sort and deduplicate the drugs...")
        # sort and deduplicate
        drugs_df.sort_values(by=["score", "num_of_shared_genes_in_path"], ascending=[False, False], inplace=True)
        drugs_df.drop_duplicates(subset=["drug_id"], inplace=True)
        
        drugs_df["pvalue"] = drugs_df["pvalue"].apply(lambda x: f"{x:.3g}")

        self.logger.info("Save the results...")
        # save the results
        lib.save_df(drugs_df, output_file, "network_annotations")
        self.logger.info(f"Network annotations saved to {output_file}")

    def merge_annotations(self, pred_xlsx, shared_genes_xlsx, shared_diseases_xlsx, network_anno_xlsx, output_file):
        """
        Merge all annotations, generate the final annotated_drugs.xlsx.
        """
        if not output_file.endswith(".xlsx"):
            raise ValueError("Output file must be an Excel file")

        # read the basic predicted results
        pred_df = pd.read_excel(pred_xlsx, sheet_name="predicted_drugs")

        # read the network annotation results (contains most fields)
        network_df = pd.read_excel(network_anno_xlsx, sheet_name="network_annotations")

        # read the shared genes annotation
        shared_genes_df = pd.read_excel(shared_genes_xlsx, sheet_name="shared_genes")

        # read the shared diseases annotation
        shared_diseases_df = pd.read_excel(shared_diseases_xlsx, sheet_name="shared_diseases")

        # merge the network annotation with other annotations
        final_df = network_df.copy()

        # merge the shared genes information
        if not shared_genes_df.empty:
            # safely handle the shared gene names, filter out NaN values
            def safe_join_gene_names(x):
                # filter out NaN values and convert to string
                valid_names = [str(name) for name in x if pd.notna(name) and str(name).strip()]
                if not valid_names:
                    return ""
                # merge all names and deduplicate
                all_names = []
                for name_str in valid_names:
                    if "|" in name_str:
                        all_names.extend(name_str.split("|"))
                    else:
                        all_names.append(name_str)
                # deduplicate and filter out empty strings
                unique_names = list(set([name.strip() for name in all_names if name.strip()]))
                return "|".join(unique_names)

            shared_genes_summary = shared_genes_df.groupby("predicted_drug_id").agg({
                "num_of_shared_genes": "sum",
                "shared_gene_names": safe_join_gene_names
            }).reset_index()
            shared_genes_summary.rename(columns={"predicted_drug_id": "drug_id"}, inplace=True)
            final_df = final_df.merge(shared_genes_summary, on="drug_id", how="left")
            final_df["num_of_shared_genes"] = final_df["num_of_shared_genes"].fillna(0)
            final_df["shared_gene_names"] = final_df["shared_gene_names"].fillna("")
        else:
            final_df["num_of_shared_genes"] = 0
            final_df["shared_gene_names"] = ""

        # merge the shared diseases information
        if not shared_diseases_df.empty:
            # safely handle the shared disease names, filter out NaN values
            def safe_join_disease_names(x):
                # filter out NaN values and convert to string
                valid_names = [str(name) for name in x if pd.notna(name) and str(name).strip()]
                if not valid_names:
                    return ""
                # merge all names and deduplicate
                all_names = []
                for name_str in valid_names:
                    if "|" in name_str:
                        all_names.extend(name_str.split("|"))
                    else:
                        all_names.append(name_str)
                # deduplicate and filter out empty strings
                unique_names = list(set([name.strip() for name in all_names if name.strip()]))
                return "|".join(unique_names)

            shared_diseases_summary = shared_diseases_df.groupby("drug_id").agg({
                "num_of_shared_diseases": "sum",
                "shared_disease_names": safe_join_disease_names
            }).reset_index()
            final_df = final_df.merge(shared_diseases_summary, on="drug_id", how="left")
            final_df["num_of_shared_diseases"] = final_df["num_of_shared_diseases"].fillna(0)
            final_df["shared_disease_names"] = final_df["shared_disease_names"].fillna("")
        else:
            final_df["num_of_shared_diseases"] = 0
            final_df["shared_disease_names"] = ""

        # ensure all necessary fields exist
        required_fields = [
            "drug_id", "drug_name", "pvalue", "rank", "score", "num_of_shared_genes_in_path", "paths", 
            "existing", "num_of_shared_pathways", "key_genes", "drug_degree",
            "num_of_shared_genes", "shared_gene_names", "num_of_shared_diseases", 
            "shared_disease_names", "num_of_key_genes", "shared_pathways"
        ]

        for field in required_fields:
            if field not in final_df.columns:
                final_df[field] = ""

        # rearrange the column order, ensure it is consistent with the final table in run.ipynb
        column_order = [
            "drug_id", "drug_name", "pvalue", "rank", "score", "num_of_shared_genes_in_path", "paths", 
            "existing", "num_of_shared_pathways", "shared_pathways", "key_genes", 
            "num_of_key_genes", "drug_degree", "num_of_shared_genes", 
            "shared_gene_names", "num_of_shared_diseases", "shared_disease_names"
        ]

        # only keep the existing columns
        available_columns = [col for col in column_order if col in final_df.columns]
        final_df = final_df[available_columns]

        # fill the missing values
        final_df = final_df.fillna({
            "num_of_shared_genes_in_path": 0,
            "paths": "",
            "existing": False,
            "num_of_shared_pathways": 0,
            "shared_pathways": "",
            "key_genes": "",
            "num_of_key_genes": 0,
            "drug_degree": 0,
            "num_of_shared_genes": 0,
            "shared_gene_names": "",
            "num_of_shared_diseases": 0,
            "shared_disease_names": ""
        })

        # ensure the data types are correct
        final_df["num_of_shared_genes_in_path"] = final_df["num_of_shared_genes_in_path"].astype(int)
        final_df["existing"] = final_df["existing"].astype(bool)
        final_df["num_of_shared_pathways"] = final_df["num_of_shared_pathways"].astype(int)
        final_df["num_of_key_genes"] = final_df["num_of_key_genes"].astype(int)
        final_df["drug_degree"] = final_df["drug_degree"].astype(int)
        final_df["num_of_shared_genes"] = final_df["num_of_shared_genes"].astype(int)
        final_df["num_of_shared_diseases"] = final_df["num_of_shared_diseases"].astype(int)

        # sort by score and shared gene counts
        final_df.sort_values(by=["score", "num_of_shared_genes_in_path"], ascending=[False, False], inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        # save the final results
        lib.save_df(final_df, output_file, "annotated_drugs")
        self.logger.info(f"Final annotated drugs saved to {output_file}")

        return final_df

    def shared_diseases(self, *args, **kwargs):
        """
        Count the shared diseases of drugs and diseases.
        """
        pass

    def shared_genes_pathways(self, *args, **kwargs):
        """
        Count the shared genes and pathways of drugs and diseases.
        """
        pass 
