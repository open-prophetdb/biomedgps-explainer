import pandas as pd
import numpy as np
import os
from .. import lib
import networkx as nx
from gseapy import enrichr

class DrugDiseaseCore:
    """
    负责潜在药物列表生成与主数据处理逻辑。
    """
    def __init__(self):
        self.logger = lib.get_logger("DrugDiseaseCore")

    def run_full_pipeline(self, disease_id, entity_file, knowledge_graph, entity_embeddings, relation_embeddings, output_dir, model='TransE_l2', top_n_diseases=100, gamma=12.0, threshold=0.5, relation_type='GNBR::T::Compound:Disease'):
        """
        一键完成所有分析步骤，输出 annotated_drugs.xlsx，字段与 run.ipynb 最终表一致。
        """
        # 1. 药物预测
        pred_xlsx = os.path.join(output_dir, 'predicted_drugs.xlsx')
        self.predict_drugs(
            disease_id, entity_file, knowledge_graph, entity_embeddings, relation_embeddings,
            model, top_n_diseases, gamma, threshold, relation_type, pred_xlsx
        )
        # 2. shared genes/pathways 注释
        shared_genes_xlsx = os.path.join(output_dir, 'shared_genes_pathways.xlsx')
        self.annotate_shared_genes_pathways(
            pred_xlsx, disease_id, knowledge_graph, shared_genes_xlsx
        )
        # 3. shared diseases 注释
        shared_diseases_xlsx = os.path.join(output_dir, 'shared_diseases.xlsx')
        self.annotate_shared_diseases(
            pred_xlsx, disease_id, knowledge_graph, entity_embeddings, relation_embeddings, shared_diseases_xlsx, model, gamma, top_n_diseases
        )
        # 4. 网络分析/中心性/路径等
        network_anno_xlsx = os.path.join(output_dir, 'network_annotations.xlsx')
        self.annotate_network_features(
            pred_xlsx, disease_id, knowledge_graph, entity_file, network_anno_xlsx
        )
        # 5. 整合所有注释，输出 annotated_drugs.xlsx
        annotated_xlsx = os.path.join(output_dir, 'annotated_drugs.xlsx')
        self.merge_annotations(
            pred_xlsx, shared_genes_xlsx, shared_diseases_xlsx, network_anno_xlsx, annotated_xlsx
        )
        self.logger.info(f'已生成 {annotated_xlsx}')

    def predict_drugs(self, disease_id, entity_file, knowledge_graph, entity_embeddings, relation_embeddings, model, top_n_diseases, gamma, threshold, relation_type, output_file):
        """
        生成潜在药物列表，保存 annotated_drugs.xlsx。
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

        # BioMedGPS格式的知识图谱已经包含了source_name和target_name
        if not set(["source_name", "target_name"]).issubset(set(kg_df.columns)):
            # 如果缺少名称列，从annotated_entities.tsv加载
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
        # BioMedGPS格式的embedding列已经是字符串格式的向量
        entity_embeddings_df["embedding"] = entity_embeddings_df["embedding"].apply(lambda x: np.array(x.split("|"), dtype=float))
        self.logger.info("Entity embeddings file is loaded")

        # Load relation embeddings (BioMedGPS format)
        relation_embeddings_df = pd.read_csv(relation_embeddings, sep="\t")
        # BioMedGPS格式的embedding列已经是字符串格式的向量
        relation_embeddings_df["embedding"] = relation_embeddings_df["embedding"].apply(lambda x: np.array(x.split("|"), dtype=float))
        self.logger.info("Relation embeddings file is loaded")

        if disease_id not in entity_embeddings_df[entity_embeddings_df["entity_type"] == "Disease"]["entity_id"].to_numpy():
            raise ValueError("The disease id is not in the entity file.")

        target_disease_embedding = entity_embeddings_df[(entity_embeddings_df["entity_id"] == disease_id) & (entity_embeddings_df["entity_type"] == "Disease")]["embedding"].to_numpy()[0]
        
        # BioMedGPS格式的关系ID查找
        relation_embedding = None
        if "id" in relation_embeddings_df.columns:
            relation_embedding = relation_embeddings_df[relation_embeddings_df["id"] == relation_type]["embedding"].to_numpy()[0]
        elif "relation_type" in relation_embeddings_df.columns:
            relation_embedding = relation_embeddings_df[relation_embeddings_df["relation_type"] == relation_type]["embedding"].to_numpy()[0]
        else:
            # 如果找不到关系嵌入，使用默认向量
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

    def annotate_shared_genes_pathways(self, predicted_drug_file, disease_id, knowledge_graph, output_file):
        """
        统计药物与疾病的共享基因和通路，输出 shared_genes_pathways.xlsx。
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

        predicted_df = pd.read_csv(predicted_drug_file, sep="\t", dtype=str)
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
        shared_genes_df = pd.DataFrame({
            "known_disease_id": np.repeat(known_df["id"], len(predicted_df["drug_id"].unique())),
            "known_disease_name": np.repeat(known_df["name"], len(predicted_df["drug_name"].unique())),
            "predicted_drug_id": predicted_df["drug_id"].tolist() * len(known_df["id"].unique()),
            "predicted_drug_name": predicted_df["drug_name"].tolist() * len(known_df["name"].unique()),
            "shared_genes": ["|".join(x) for x in get_shared_genes()],
            "shared_gene_names": ["|".join(x) for x in get_shared_gene_names()],
            "number_of_shared_genes": [len(x) for x in get_shared_genes()],
        })
        self.logger.info("Shared genes dataframe is created")
        lib.save_df(shared_genes_df, output_file, "shared_genes")

        self.logger.info("Performing pathway enrichment analysis")
        known_disease_enrichment_results = []
        for i, row in known_df.iterrows():
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
        for i, row in predicted_df.iterrows():
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
        统计药物与疾病的共享疾病，输出 shared_diseases.xlsx。
        """
        if not output_file.endswith(".xlsx"):
            raise ValueError("Output file must be an Excel file")

        similar_disease_relation_type = "BioMedGPS::SimilarWith::Disease:Disease"

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

        relation_embedding = relations_df[relations_df["id"] == similar_disease_relation_type]["embedding"].to_numpy()[0]

        predicted_df = pd.read_csv(predicted_drug_file, sep="\t")
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
        disease_embedding = list(entities_df[entities_df["entity_id"] == disease_id]["embedding"].to_numpy()[0])
        all_disease_embeddings = list(entities_df[entities_df["entity_type"] == "Disease"]["embedding"].to_numpy())
        all_disease_ids = entities_df[entities_df["entity_type"] == "Disease"]["entity_id"].to_numpy()

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
        for index, drug in enumerate(predicted_drug_ids):
            treated_disease_ids = kg_df[
                (kg_df["source_id"] == drug) & (kg_df["relation_type"].isin(["BioMedGPS::Treatment::Compound:Disease"]))
            ]["target_id"].to_numpy()

            shared_disease_ids = np.intersect1d(treated_disease_ids, top_n_disease_ids)
            shared_disease_names = kg_df[kg_df["target_id"].isin(shared_disease_ids)]["target_name"].to_numpy()

            results.append({
                "drug_id": drug,
                "drug_name": predicted_drug_names[index],
                "num_of_treated_diseases": len(treated_disease_ids),
                "num_of_shared_diseases": len(shared_disease_ids),
                "shared_disease_names": "|".join(list(set(shared_disease_names))),
            })

        self.logger.info("Stat the shared diseases")
        results_df = pd.DataFrame(results)
        lib.save_df(results_df, output_file, "shared_diseases")
        self.logger.info("The results are saved to the file")

    def annotate_network_features(self, predicted_drug_file, disease_id, knowledge_graph, entity_file, output_file):
        """
        计算药物-疾病-基因路径、中心性、PPI、通路富集等网络注释，输出 network_annotations.xlsx。
        """
        if not output_file.endswith(".xlsx"):
            raise ValueError("Output file must be an Excel file")

        # 读取知识图谱
        kg_df = pd.read_csv(knowledge_graph, sep="\t")
        kg_df["source"] = kg_df["source_type"] + "::" + kg_df["source_id"]
        kg_df["target"] = kg_df["target_type"] + "::" + kg_df["target_id"]

        # 读取实体文件
        entities_df = pd.read_csv(entity_file, sep="\t")
        entities_df["id"] = entities_df["label"] + "::" + entities_df["id"]
        entities_df["xrefs"] = entities_df["xrefs"].astype(str)
        entities_df["symbol"] = entities_df["xrefs"].apply(
            lambda x: next((item.split(":")[1] for item in x.split("|") if item.startswith("SYMBOL:")), None)
        )
        entities_df["symbol"] = entities_df["symbol"].fillna(entities_df["name"])
        id_to_name = dict(zip(entities_df["id"], entities_df["symbol"]))

        # 读取预测的药物
        drugs_df = pd.read_excel(predicted_drug_file, sheet_name="predicted_drugs")
        drugs_df["id"] = "Compound::" + drugs_df["drug_id"]
        all_drugs = drugs_df["id"].tolist()

        # 目标疾病
        target_disease = f"Disease::{disease_id}"
        allowed_types = {"Disease", "Gene", "Pathway"}

        # 创建有向图
        G = nx.DiGraph()
        for _, row in kg_df.iterrows():
            G.add_edge(row["source"], row["target"], relation=row["relation_type"])

        # 存储药物到疾病的所有两跳路径
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

        # 解析路径数据
        formatted_results = []
        for drug, paths in drug_disease_paths.items():
            drug_name = id_to_name.get(drug, drug)
            for path in paths:
                path_names = [id_to_name.get(node_id, node_id) for node_id in path]
                formatted_results.append([drug.split("::")[1], drug_name, path_names[1], path_names[2], disease_id])

        df_results = pd.DataFrame(formatted_results, columns=["drug_id", "drug_name", "path", "disease_name", "disease_id"])

        # 统计路径数量
        df = df_results.groupby(["drug_id", "disease_id"]).size().reset_index(name="path_count")
        drugs_df["shared_gene_counts"] = drugs_df["drug_id"].map(df.set_index("drug_id")["path_count"])
        drugs_df["shared_gene_counts"] = drugs_df["shared_gene_counts"].fillna(0).astype(int)
        
        df = df_results.groupby(["drug_id", "disease_id"]).agg({"path": lambda x: ";".join(x)}).reset_index()
        drugs_df["paths"] = drugs_df["drug_id"].map(df.set_index("drug_id")["path"])

        # 检查现有药物
        existing_drugs = kg_df[
            ((kg_df["target"] == target_disease) & (kg_df["source_type"] == "Compound")) |
            ((kg_df["source"] == target_disease) & (kg_df["target_type"] == "Compound"))
        ]
        existing_drugs = existing_drugs["source_id"].tolist() + existing_drugs["target_id"].tolist()
        existing_drugs = list(set(existing_drugs))
        drugs_df["existing"] = drugs_df["drug_id"].isin(existing_drugs)

        # 通路富集分析
        specific_disease_gene = kg_df[
            ((kg_df["source_id"] == disease_id) & (kg_df["target_type"] == "Gene")) |
            ((kg_df["target_id"] == disease_id) & (kg_df["source_type"] == "Gene"))
        ]
        specific_disease_genes = set(
            specific_disease_gene["target_id"][specific_disease_gene["source_id"] == disease_id]
        ).union(set(specific_disease_gene["source_id"][specific_disease_gene["target_id"] == disease_id]))

        # 提取药物调控的基因关系
        drug_gene_edges = kg_df[
            ((kg_df["source_type"] == "Compound") & (kg_df["target_type"] == "Gene")) |
            ((kg_df["source_type"] == "Gene") & (kg_df["target_type"] == "Compound"))
        ]

        # 疾病通路富集
        specific_disease_gene_symbols = [id_to_name.get("Gene::" + gene, gene) for gene in specific_disease_genes]
        enrich_results = enrichr(
            gene_list=list(specific_disease_gene_symbols),
            gene_sets=["KEGG_2021_Human"],
            outdir=None,
        )
        enrich_df = enrich_results.results
        enrich_df = enrich_df[enrich_df["Adjusted P-value"] < 0.05]

        # 药物通路富集和重叠分析
        for index, drug in drugs_df.head(1000).iterrows():
            drug_id, drug_name = drug["drug_id"], drug["drug_name"]
            
            # 药物调控的基因列表
            regulated_genes = set(
                drug_gene_edges["target_id"][drug_gene_edges["source_id"] == drug_id]
            ).union(set(drug_gene_edges["source_id"][drug_gene_edges["target_id"] == drug_id]))
            
            regulated_gene_symbols = [id_to_name.get("Gene::" + gene, gene) for gene in regulated_genes]
            
            if len(regulated_gene_symbols) < 5:
                continue
                
            # 药物通路富集
            drug_enrich_results = enrichr(
                gene_list=list(regulated_gene_symbols),
                gene_sets=["KEGG_2021_Human"],
                outdir=None,
            )
            drug_enrich_df = drug_enrich_results.results
            drug_enrich_df = drug_enrich_df[drug_enrich_df["Adjusted P-value"] < 0.05]
            
            # 计算重叠通路
            overlap_pathways = set(drug_enrich_df["Term"]) & set(enrich_df["Term"])
            overlap_pathways = list(overlap_pathways)
            overlap_pathways.sort()
            drugs_df.loc[index, "overlap_pathways_count"] = len(overlap_pathways)
            drugs_df.loc[index, "overlap_pathways"] = ";".join(overlap_pathways)

        # PPI网络分析
        gene_gene_edges = kg_df[(kg_df["source_type"] == "Gene") & (kg_df["target_type"] == "Gene")]
        ppi_network = nx.from_pandas_edgelist(gene_gene_edges, "source_id", "target_id")

        for index, drug in drugs_df.head(1500).iterrows():
            drug_id, drug_name = drug["drug_id"], drug["drug_name"]
            
            # 药物调控基因
            regulated_genes = set(
                kg_df["target_id"][(kg_df["source_id"] == drug_id) & (kg_df["target_type"] == "Gene")]
            ).union(set(kg_df["source_id"][(kg_df["target_id"] == drug_id) & (kg_df["source_type"] == "Gene")]))
            
            # 提取药物调控基因子网络
            subgraph_genes = [gene for gene in regulated_genes if gene in ppi_network]
            subgraph = ppi_network.subgraph(subgraph_genes)
            
            if subgraph.number_of_edges() == 0:
                drugs_df.loc[index, "num_key_genes"] = 0
                drugs_df.loc[index, "key_genes"] = ""
                continue

            # 计算网络中心性指标
            degree_centrality = nx.degree_centrality(subgraph)
            betweenness_centrality = nx.betweenness_centrality(subgraph)
            
            centrality_df = pd.DataFrame({
                "gene_id": list(degree_centrality.keys()),
                "degree_centrality": list(degree_centrality.values()),
                "betweenness_centrality": list(betweenness_centrality.values()),
            })
            
            centrality_df.sort_values(["degree_centrality", "betweenness_centrality"], ascending=False, inplace=True)
            centrality_df["gene_name"] = centrality_df["gene_id"].apply(lambda x: id_to_name.get("Gene::" + x, x))
            
            # 只保留与疾病相关的基因
            centrality_df = centrality_df[centrality_df["gene_id"].isin(specific_disease_genes)]
            
            if centrality_df.empty:
                drugs_df.loc[index, "num_key_genes"] = 0
                drugs_df.loc[index, "key_genes"] = ""
                continue

            # 确定关键基因（前10%）
            threshold = centrality_df["degree_centrality"].quantile(0.90)
            centrality_df = centrality_df[centrality_df["degree_centrality"] >= threshold]
            
            threshold = centrality_df["betweenness_centrality"].quantile(0.90)
            centrality_df = centrality_df[centrality_df["betweenness_centrality"] >= threshold]
            
            num_key_genes = len(centrality_df)
            drugs_df.loc[index, "num_key_genes"] = num_key_genes
            drugs_df.loc[index, "key_genes"] = ";".join(centrality_df["gene_name"].tolist())

        # 计算药物连接度
        for index, drug in drugs_df.head(1500).iterrows():
            drug_id = drug["drug_id"]
            drug_degree = kg_df[(kg_df["source_id"] == drug_id) | (kg_df["target_id"] == drug_id)]
            drugs_df.loc[index, "drug_degree"] = drug_degree.shape[0]

        # 排序和去重
        drugs_df.sort_values(by=["score", "shared_gene_counts"], ascending=[False, False], inplace=True)
        drugs_df.drop_duplicates(subset=["drug_id"], inplace=True)

        # 保存结果
        lib.save_df(drugs_df, output_file, "network_annotations")
        self.logger.info(f"Network annotations saved to {output_file}")

    def merge_annotations(self, pred_xlsx, shared_genes_xlsx, shared_diseases_xlsx, network_anno_xlsx, output_file):
        """
        整合所有注释，生成最终 annotated_drugs.xlsx。
        """
        if not output_file.endswith(".xlsx"):
            raise ValueError("Output file must be an Excel file")

        # 读取基础预测结果
        pred_df = pd.read_excel(pred_xlsx, sheet_name="predicted_drugs")
        
        # 读取网络注释结果（包含大部分字段）
        network_df = pd.read_excel(network_anno_xlsx, sheet_name="network_annotations")
        
        # 读取共享基因注释
        shared_genes_df = pd.read_excel(shared_genes_xlsx, sheet_name="shared_genes")
        
        # 读取共享疾病注释
        shared_diseases_df = pd.read_excel(shared_diseases_xlsx, sheet_name="shared_diseases")
        
        # 以网络注释为基础，合并其他注释
        final_df = network_df.copy()
        
        # 合并共享基因信息
        if not shared_genes_df.empty:
            shared_genes_summary = shared_genes_df.groupby("predicted_drug_id").agg({
                "number_of_shared_genes": "sum",
                "shared_gene_names": lambda x: "|".join(set("|".join(x).split("|")))
            }).reset_index()
            shared_genes_summary.rename(columns={"predicted_drug_id": "drug_id"}, inplace=True)
            final_df = final_df.merge(shared_genes_summary, on="drug_id", how="left")
            final_df["number_of_shared_genes"] = final_df["number_of_shared_genes"].fillna(0)
            final_df["shared_gene_names"] = final_df["shared_gene_names"].fillna("")
        else:
            final_df["number_of_shared_genes"] = 0
            final_df["shared_gene_names"] = ""
        
        # 合并共享疾病信息
        if not shared_diseases_df.empty:
            shared_diseases_summary = shared_diseases_df.groupby("drug_id").agg({
                "num_of_shared_diseases": "sum",
                "shared_disease_names": lambda x: "|".join(set("|".join(x).split("|")))
            }).reset_index()
            final_df = final_df.merge(shared_diseases_summary, on="drug_id", how="left")
            final_df["num_of_shared_diseases"] = final_df["num_of_shared_diseases"].fillna(0)
            final_df["shared_disease_names"] = final_df["shared_disease_names"].fillna("")
        else:
            final_df["num_of_shared_diseases"] = 0
            final_df["shared_disease_names"] = ""
        
        # 确保所有必要字段存在
        required_fields = [
            "drug_id", "drug_name", "score", "shared_gene_counts", "paths", 
            "existing", "overlap_pathways_count", "key_genes", "drug_degree",
            "number_of_shared_genes", "shared_gene_names", "num_of_shared_diseases", 
            "shared_disease_names", "num_key_genes", "overlap_pathways"
        ]
        
        for field in required_fields:
            if field not in final_df.columns:
                final_df[field] = ""
        
        # 重新排列列顺序，确保与 run.ipynb 最终表一致
        column_order = [
            "drug_id", "drug_name", "score", "shared_gene_counts", "paths", 
            "existing", "overlap_pathways_count", "overlap_pathways", "key_genes", 
            "num_key_genes", "drug_degree", "number_of_shared_genes", 
            "shared_gene_names", "num_of_shared_diseases", "shared_disease_names"
        ]
        
        # 只保留存在的列
        available_columns = [col for col in column_order if col in final_df.columns]
        final_df = final_df[available_columns]
        
        # 填充缺失值
        final_df = final_df.fillna({
            "shared_gene_counts": 0,
            "paths": "",
            "existing": False,
            "overlap_pathways_count": 0,
            "overlap_pathways": "",
            "key_genes": "",
            "num_key_genes": 0,
            "drug_degree": 0,
            "number_of_shared_genes": 0,
            "shared_gene_names": "",
            "num_of_shared_diseases": 0,
            "shared_disease_names": ""
        })
        
        # 确保数据类型正确
        final_df["shared_gene_counts"] = final_df["shared_gene_counts"].astype(int)
        final_df["existing"] = final_df["existing"].astype(bool)
        final_df["overlap_pathways_count"] = final_df["overlap_pathways_count"].astype(int)
        final_df["num_key_genes"] = final_df["num_key_genes"].astype(int)
        final_df["drug_degree"] = final_df["drug_degree"].astype(int)
        final_df["number_of_shared_genes"] = final_df["number_of_shared_genes"].astype(int)
        final_df["num_of_shared_diseases"] = final_df["num_of_shared_diseases"].astype(int)
        
        # 按分数和共享基因数排序
        final_df.sort_values(by=["score", "shared_gene_counts"], ascending=[False, False], inplace=True)
        final_df.reset_index(drop=True, inplace=True)
        
        # 保存最终结果
        lib.save_df(final_df, output_file, "annotated_drugs")
        self.logger.info(f"Final annotated drugs saved to {output_file}")
        
        return final_df

    def shared_diseases(self, *args, **kwargs):
        """
        统计药物与疾病的共享疾病。
        """
        pass

    def shared_genes_pathways(self, *args, **kwargs):
        """
        统计药物与疾病的共享基因和通路。
        """
        pass 