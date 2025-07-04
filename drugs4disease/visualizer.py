import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from .explain import DrugExplain


class Visualizer:
    """
    Provide common visualization and interpretation for drug discovery analysis.
    """

    def __init__(
        self,
        disease_id: str,
        disease_name: str,
        embed_images: bool = True,
    ):
        """
        Initialize the Visualizer.

        Args:
            embed_images: Whether to embed images in the HTML report.
                         If True, images will be embedded as base64 data.
                         If False, images will be referenced by file paths.
                         Default is True.
            disease_id: Disease id.
            disease_name: Disease name.
        """
        self.embed_images = embed_images
        # 设置中文字体支持
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        self.disease_id = disease_id
        self.disease_name = disease_name

    @staticmethod
    def get_chart_types():
        return [
            "score_distribution",
            "predicted_score_boxplot",
            "disease_similarity_heatmap",
            "network_centrality",
            "shared_genes_pathways",
            "drug_disease_network",
            "shared_gene_count",
            "score_vs_degree",
            "shared_gene_count_vs_score",
            "overlap_pathways",
            "key_genes_distribution",
            "existing_vs_predicted",
            "prompt",
        ]

    @staticmethod
    def get_chart_title(chart_type):
        return {
            "score_distribution": "Drug Predicted Score Distribution",
            "predicted_score_boxplot": "Predicted Score Distribution by Knowledge Graph Inclusion",
            "disease_similarity_heatmap": "Drug Similarity Heatmap",
            "network_centrality": "Network Centrality Analysis",
            "shared_genes_pathways": "Shared Genes and Pathways Analysis",
            "drug_disease_network": "Drug Similarity Network",
            "shared_gene_count": "Number of Shared Genes Between Drugs and Diseases",
            "shared_gene_count_vs_score": "Shared Gene Count vs Predicted Score",
            "score_vs_degree": "Drug Degree vs Predicted Score",
            "overlap_pathways": "Number of Overlapping Pathways Between Drugs and Diseases",
            "key_genes_distribution": "Number of Key Genes Distribution",
            "existing_vs_predicted": "Ratio of Known Drugs to Predicted Drugs",
            "prompt": "Prompt for ChatGPT Deep Research",
        }.get(chart_type, chart_type)

    def plot_prompt(
        self, df: pd.DataFrame, filtered_df: pd.DataFrame, output_file: str
    ):
        # step 4: explain the drugs
        print("\n4. Explain the drugs...")
        drug_names = filtered_df["drug_name"].tolist()[:50]
        explainer = DrugExplain()
        prompt = explainer.generate_prompt(
            drug_names=drug_names, disease_name=self.disease_name
        )
        with open(output_file, "w") as f:
            f.write(prompt)

        interpretation = "This section provides a prompt for ChatGPT Deep Research to do the literature research on the drugs in relation to the disease."
        return interpretation

    def create_visualization(
        self,
        data_file: str,
        viz_type: str,
        output_file: str,
        sheet_names: tuple[str, str] = ("annotated_drugs", "filtered_drugs"),
    ) -> str:
        """
        Generate charts based on specified visualization types.

        Args:
            data_file: Path to the input data file
            viz_type: Visualization type
            output_file: Path to the output file
            sheet_names: Tuple of sheet names for the input data file. Default is ("annotated_drugs", "filtered_drugs").

        Returns:
            Interpretation of the chart
        """
        # Read data
        df = pd.read_excel(data_file, sheet_name=sheet_names[0])
        filtered_df = pd.read_excel(data_file, sheet_name=sheet_names[1])

        if viz_type not in self.get_chart_types():
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        else:
            print(f"Generating {viz_type} chart...")

        func = getattr(self, f"plot_{viz_type}")
        return func(df, filtered_df, output_file)

    def generate_report(
        self,
        data_file: str,
        output_file: str,
        title: str = "Drug Repurposing Analysis Report",
        sheet_names: tuple[str, str] = ("annotated_drugs", "filtered_drugs"),
    ) -> str:
        """
        Generate comprehensive HTML report

        Args:
            data_file: Path to the input data file
            output_file: Path to the output HTML file
            title: Report title
            sheet_names: Tuple of sheet names for the input data file. Default is ("annotated_drugs", "filtered_drugs").

        Returns:
            Status information of the report generation
        """
        # Read data
        df = pd.read_excel(data_file, sheet_name=sheet_names[0])
        filtered_df = pd.read_excel(data_file, sheet_name=sheet_names[1])

        # Generate report directory
        report_dir = os.path.dirname(output_file)
        os.makedirs(report_dir, exist_ok=True)

        # 生成所有图表
        charts = {}
        chart_types = self.get_chart_types()

        for chart_type in chart_types:
            try:
                chart_file = os.path.join(report_dir, f"{chart_type}.png")
                json_file = os.path.join(report_dir, f"{chart_type}.json")

                interpretation = self.create_visualization(
                    data_file=data_file,
                    viz_type=chart_type,
                    output_file=chart_file,
                    sheet_names=("annotated_drugs", "filtered_drugs"),
                )

                if os.path.exists(json_file):
                    chart_file = json_file
                else:
                    # 如果不需要内嵌图片，使用相对路径
                    if not self.embed_images:
                        chart_file = os.path.basename(chart_file)
                    else:
                        chart_file = chart_file

                charts[chart_type] = {
                    "file": chart_file,
                    "interpretation": interpretation,
                }
            except Exception as e:
                print(f"Failed to generate chart {chart_type}: {e}")

        # Generate HTML report
        html_content = self._generate_html_report(df, filtered_df, charts, title)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return f"Report generated: {output_file}"

    @staticmethod
    def render_table(df: pd.DataFrame, table_id: str) -> str:
        """Plot the table of the filtered drugs"""
        all_columns = df.columns.tolist()
        print("All columns in the table: ", all_columns)

        column_rename = {
            "drug_id": "ID",
            "drug_name": "Name",
            "score": "Score",
            "num_of_shared_genes_in_path": "# Shared Genes",
            "existing": "Existing",
            "num_of_shared_pathways": "# Overlap Pathways",
            "key_genes": "Key Genes",
            "num_of_key_genes": "# Key Genes",
            "drug_degree": "Degree",
            "num_of_shared_genes": "# Shared Genes",
            "num_of_shared_diseases": "# Shared Diseases",
            "shared_disease_names": "Shared Disease Names",
            "paths": "Paths",
            "shared_gene_names": "Shared Gene Names",
            "shared_pathways": "Overlap Pathways",
        }

        df_renamed = df.rename(columns=column_rename)

        return df_renamed.to_html(
            index=False,
            border=0,
            classes="display nowrap",
            table_id=table_id,
            formatters={
                "Score": lambda x: f"{x:.3f}",
                "Existing": lambda x: "Yes" if x else "No",
            },
        )

    @staticmethod
    def plot_score_distribution(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        import plotly.express as px

        # Plot the predicted score distribution
        fig = px.histogram(
            data_frame=df,
            x="score",
            nbins=30,
            marginal="box",
            opacity=0.75,
            title="Predicted Score Distribution",
        )

        # Update the layout
        fig.update_layout(
            xaxis_title="Predicted Score",
            yaxis_title="Drug Count",
            bargap=0.1,
            plot_bgcolor="white",
            xaxis=dict(
                showline=True,
                linecolor="black",
                dtick=0.05,
                range=[0, 1],
            ),
            yaxis=dict(showline=True, linecolor="black"),
            height=600,
        )

        fig.write_image(output_path)

        # Generate plotly json file and load it in HTML
        json_path = output_path.replace(".png", ".json")
        fig.write_json(json_path)

        # Interpretation
        interpretation = """#### Predicted Score Distribution of Candidate Drugs

Figure: Predicted Score Distribution of Candidate Drugs

This plot illustrates the distribution of predicted scores assigned by the Graph Neural Network (GNN) model to candidate drugs in relation to the disease.

- The histogram (bottom) shows the number of drugs falling into specific predicted score ranges.

- The box plot (top) summarizes the overall distribution, highlighting the interquartile range (IQR), median, and outliers.

We observe that:

- Most drugs receive relatively low predicted scores, suggesting limited therapeutic relevance.

- A smaller subset of drugs falls into the upper tail, indicating higher potential for anti-scarring efficacy.

- The model effectively distinguishes a few high-confidence candidates, which are worth prioritizing for further validation or analysis.
        """

        return interpretation

    @staticmethod
    def plot_predicted_score_boxplot(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        """Plot the predicted score boxplot"""
        import plotly.express as px
        import pandas as pd

        # 构建 drug_id 的 set，用于高效匹配
        new_candidate_ids = set(
            filtered_df[~filtered_df["existing"]]["drug_id"].astype(str)
        )
        num_existing_drugs = df["existing"].value_counts().get(True, 0)
        num_new_candidates = filtered_df[~filtered_df["existing"]].shape[0]
        num_others = df.shape[0] - num_existing_drugs - num_new_candidates

        # 添加 'group' 列
        def classify_group(row):
            drug_id = str(row["drug_id"])
            if row["existing"]:
                return f"Existing drugs in Influenza ({num_existing_drugs})"
            elif drug_id in new_candidate_ids:
                return f"Predicted new candidates ({num_new_candidates})"
            else:
                return f"All other drugs ({num_others})"

        df["group"] = df.apply(classify_group, axis=1)
        df["formatted_drug_name"] = df["drug_name"].str.title()

        # 绘制箱线图
        fig = px.box(
            df,
            x="group",
            y="score",
            points="outliers",
            color="group",
            color_discrete_map={
                f"Existing drugs in the disease ({num_existing_drugs})": "blue",
                f"Predicted new candidates ({num_new_candidates})": "red",
                f"All other drugs ({num_others})": "lightgrey",
            },
            title="Predicted Score Distribution for Drug Candidates",
            labels={"score": "Predicted Score", "group": ""},
        )

        # 更新布局为纵向 portrait
        fig.update_layout(
            plot_bgcolor="white",
            height=940,
            width=800,
            font=dict(size=28),
            xaxis=dict(showline=True, linecolor="black", tickangle=20),
            yaxis=dict(showline=True, linecolor="black", title="Predicted Score"),
            showlegend=False,
            title=None,
        )

        fig.write_image(output_path, width=800, height=940, scale=2)

        # Generate plotly json file and load it in HTML
        json_path = output_path.replace(".png", ".json")
        fig.write_json(json_path)

        interpretation = f"""#### Predicted Score Distribution by Knowledge Graph Inclusion

This box plot compares the predicted scores between drugs that exist in the knowledge graph and those that do not.

- The x-axis indicates drug presence in the biomedical knowledge graph:

  - "Existing drugs in the disease ({num_existing_drugs})"

  - "Predicted new candidates ({num_new_candidates})"

  - "All other drugs ({num_others})"

- The y-axis shows the predicted scores assigned by the GNN model.

Key observations:

- Drugs already present in the knowledge graph tend to have much higher scores, suggesting that the model successfully learns from known associations.

- In contrast, most candidate drugs not present in the graph are scored low, but a small number receive high predicted scores, representing potential novel drug-disease associations.

- This validates the model's capacity for both confirming known links and discovering promising new ones.
        """
        return interpretation

    @staticmethod
    def plot_disease_similarity_heatmap(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        """Plot the disease similarity heatmap (based on the number of shared diseases)"""
        import plotly.express as px

        # Select the top 50 drugs for analysis
        top_drugs = df.head(50).copy()

        # 获取药物名称（假设有 drug_name 列，否则请换成实际列名）
        drug_names = (
            top_drugs["drug_name"].tolist()
            if "drug_name" in top_drugs.columns
            else top_drugs.index.astype(str).tolist()
        )

        # Create the shared disease number matrix
        similarity_matrix = np.zeros((len(top_drugs), len(top_drugs)))

        for i, drug1 in top_drugs.iterrows():
            for j, drug2 in top_drugs.iterrows():
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    shared1 = drug1.get("num_of_shared_diseases", 0)
                    shared2 = drug2.get("num_of_shared_diseases", 0)
                    similarity_matrix[i][j] = (
                        min(shared1, shared2) / max(shared1, shared2)
                        if max(shared1, shared2) > 0
                        else 0
                    )

        # Convert matrix to DataFrame for plotly
        max_name_length = 10
        short_drug_names = [
            name if len(name) <= max_name_length else name[:max_name_length] + "..."
            for name in drug_names
        ]

        similarity_df = pd.DataFrame(
            similarity_matrix, index=drug_names, columns=drug_names
        )

        # Plot using plotly
        fig = px.imshow(
            similarity_df,
            color_continuous_scale="YlOrRd",
            labels=dict(x="Drug", y="Drug", color="Similarity"),
            title="Drug Similarity Heatmap (Based on Shared Diseases)",
        )

        fig.update_layout(
            xaxis_title=None,
            yaxis_title=None,
            height=900,
            title_font=dict(size=20, family="Arial", color="black"),
        )

        fig.write_image(output_path)

        # Generate plotly json file and load it in HTML
        json_path = output_path.replace(".png", ".json")
        fig.write_json(json_path)

        interpretation = (
            "This interactive heatmap shows the similarity between drugs based on the number of similar diseases they treat. "
            "The deeper the color, the more similar the drugs are. "
            "You can hover to see exact similarity values. "
            "This helps identify groups of drugs with similar mechanisms of action."
        )
        return interpretation

    @staticmethod
    def plot_network_centrality(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        """Plot the network centrality analysis chart"""
        plt.figure(figsize=(12, 8))

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))

        # 1. Drug degree distribution
        sns.histplot(df["drug_degree"], bins=20, ax=ax1, color="lightblue", alpha=0.7)
        ax1.set_title("Drug Degree Distribution", fontweight="bold")
        ax1.set_xlabel("Degree")
        ax1.set_ylabel("Drug Count")

        # 2. Number of key genes distribution
        sns.histplot(
            df["num_of_key_genes"], bins=20, ax=ax2, color="lightgreen", alpha=0.7
        )
        ax2.set_title("Number of Key Genes Distribution", fontweight="bold")
        ax2.set_xlabel("Number of Key Genes")
        ax2.set_ylabel("Drug Count")

        # 3. Drug degree vs score scatter plot
        sns.scatterplot(data=df, x="drug_degree", y="score", ax=ax3, alpha=0.6)
        ax3.set_title("Drug Degree vs Predicted Score", fontweight="bold")
        ax3.set_xlabel("Degree")
        ax3.set_ylabel("Predicted Score")

        # 4. Number of key genes vs score scatter plot
        sns.scatterplot(data=df, x="num_of_key_genes", y="score", ax=ax4, alpha=0.6)
        ax4.set_title("Number of Key Genes vs Predicted Score", fontweight="bold")
        ax4.set_xlabel("Number of Key Genes")
        ax4.set_ylabel("Predicted Score")

        # 5. Shared gene number vs score scatter plot
        sns.scatterplot(
            data=df, x="num_of_shared_genes_in_path", y="score", ax=ax5, alpha=0.6
        )
        ax5.set_title("Number of Shared Genes vs Predicted Score", fontweight="bold")
        ax5.set_xlabel("Number of Shared Genes")
        ax5.set_ylabel("Predicted Score")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretation = (
            "This chart shows the centrality features of drugs in the network. "
            "It includes the distribution of drug degree, the distribution of number of key genes, "
            "and the relationship between these features and the predicted score. "
            "Drugs with high network centrality may have stronger regulatory capabilities."
        )
        return interpretation

    @staticmethod
    def plot_shared_gene_count_vs_score(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        """Plot the shared gene count vs score scatter plot"""
        import plotly.express as px

        # 构建 drug_id 的 set，用于高效匹配
        new_candidate_ids = set(
            filtered_df[~filtered_df["existing"]]["drug_id"].astype(str)
        )
        num_existing_drugs = df["existing"].value_counts().get(True, 0)
        num_new_candidates = filtered_df[~filtered_df["existing"]].shape[0]
        num_others = df.shape[0] - num_existing_drugs - num_new_candidates

        # 添加 'group' 列
        def classify_group(row):
            drug_id = str(row["drug_id"])
            if row["existing"]:
                return f"Existing drugs in Influenza ({num_existing_drugs})"
            elif drug_id in new_candidate_ids:
                return f"Predicted new candidates ({num_new_candidates})"
            else:
                return f"All other drugs ({num_others})"

        df["group"] = df.apply(classify_group, axis=1)
        df["formatted_drug_name"] = df["drug_name"].str.title()

        # 绘图
        fig = px.scatter(
            data_frame=df,
            x="num_of_shared_genes_in_path",
            y="score",
            hover_data=["formatted_drug_name"],
            color="group",
            color_discrete_map={
                f"All other drugs ({num_others})": "lightgrey",
                f"Existing drugs in Influenza ({num_existing_drugs})": "blue",
                f"Predicted new candidates ({num_new_candidates})": "red",
            },
            title="Shared Gene Count vs Predicted Score",
        )

        # 样式调整
        fig.update_layout(
            xaxis_title="Shared Gene Count",
            yaxis_title="Predicted Score",
            plot_bgcolor="white",
            height=900,
            font=dict(size=28),
            xaxis=dict(showline=True, linecolor="black", ticks="outside"),
            yaxis=dict(showline=True, linecolor="black", ticks="outside"),
            # 👇 Legend 位置设置：图内部、下方居中
            legend=dict(
                title="Group",
                orientation="v",  # 横向排列
                yanchor="bottom",
                y=0.1,  # 控制垂直位置，负数代表图下方
                xanchor="center",
                x=0.8,  # 横向居中
            ),
            title=None,
        )
        fig.update_traces(marker=dict(size=10))
        fig.update_traces(
            selector=dict(name=f"All other drugs ({num_others})"),
            marker=dict(opacity=0.3),
        )

        fig.write_image(output_path, width=1400, height=900, scale=2)

        # Generate plotly json file and load it in HTML
        json_path = output_path.replace(".png", ".json")
        fig.write_json(json_path)

        # Interpretation
        interpretation = """
#### Shared Gene Count vs Predicted Score

This scatter plot visualizes the relationship between the number of shared genes (between each drug and the disease) and the GNN-predicted therapeutic score.

- X-axis: The count of genes shared between the drug's target profile and disease-associated genes.

- Y-axis: The predicted relevance score of the drug for treating the disease.

Key insights:

- Most candidate drugs share few or no genes with the disease, which is reflected in the dense clustering on the left.

- While a general upward trend is visible (higher shared gene count may correspond to slightly higher scores), the correlation is weak, suggesting the GNN model captures more complex features than simple gene overlap.

This visualization supports the idea that while shared gene information is relevant, it alone is not sufficient to explain the model's predictions — showcasing the added value of using a graph-based model.
        """

        return interpretation

    @staticmethod
    def plot_shared_genes_pathways(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        """Plot the shared genes and pathways analysis chart"""
        plt.figure(figsize=(15, 10))

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Shared gene number distribution
        sns.histplot(
            df["num_of_shared_genes"], bins=20, ax=ax1, color="lightcoral", alpha=0.7
        )
        ax1.set_title("Shared Gene Number Distribution", fontweight="bold")
        ax1.set_xlabel("Number of Shared Genes")
        ax1.set_ylabel("Drug Count")

        # 2. 重叠通路数量分布
        sns.histplot(
            df["num_of_shared_pathways"],
            bins=20,
            ax=ax2,
            color="lightyellow",
            alpha=0.7,
        )
        ax2.set_title("Overlapping Pathways Number Distribution", fontweight="bold")
        ax2.set_xlabel("Number of Overlapping Pathways")
        ax2.set_ylabel("Drug Count")

        # 3. 共享基因数量 vs 得分
        sns.scatterplot(data=df, x="num_of_shared_genes", y="score", ax=ax3, alpha=0.6)
        ax3.set_title("Number of Shared Genes vs Predicted Score", fontweight="bold")
        ax3.set_xlabel("Number of Shared Genes")
        ax3.set_ylabel("Predicted Score")

        # 4. 重叠通路数量 vs 得分
        sns.scatterplot(
            data=df, x="num_of_shared_pathways", y="score", ax=ax4, alpha=0.6
        )
        ax4.set_title(
            "Number of Overlapping Pathways vs Predicted Score", fontweight="bold"
        )
        ax4.set_xlabel("Number of Overlapping Pathways")
        ax4.set_ylabel("Predicted Score")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretation = (
            "This chart shows the shared genes and pathways features of drugs. "
            "Drugs with more shared genes and overlapping pathways may have better therapeutic effects, "
            "as they can regulate the biological mechanisms related to the disease."
        )
        return interpretation

    @staticmethod
    def plot_drug_disease_network(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        """Plot the drug-disease network relationship chart"""
        # Select the top 20 drugs for network visualization
        top_drugs = df.head(20)

        # Create the network graph
        import networkx as nx

        G = nx.Graph()

        # Add drug nodes
        for _, drug in top_drugs.iterrows():
            G.add_node(drug["drug_name"], type="drug", score=drug["score"])

        # Add edges based on similarity (simplified based on score similarity)
        for i, drug1 in top_drugs.iterrows():
            for j, drug2 in top_drugs.iterrows():
                if i < j:  # Avoid duplicate edges
                    # Calculate similarity (based on score difference)
                    similarity = 1 - abs(drug1["score"] - drug2["score"])
                    if similarity > 0.8:  # Only connect drugs with high similarity
                        G.add_edge(
                            drug1["drug_name"], drug2["drug_name"], weight=similarity
                        )

        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=[G.nodes[node]["score"] for node in G.nodes()],
            node_size=500,
            cmap=plt.cm.Reds,
            alpha=0.7,
            ax=ax,
        )

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)

        ax.set_title("Drug Similarity Network", fontsize=14, fontweight="bold")

        # Create colorbar and specify axes
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
        sm.set_array([G.nodes[node]["score"] for node in G.nodes()])
        cbar = plt.colorbar(sm, ax=ax, label="Predicted Score")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretation = (
            "This network graph shows the relationship between candidate drugs. "
            "The node size and color represent the predicted score of the drugs, "
            "and the edge represents the similarity between drugs. "
            "This helps identify drug clusters and potential synergistic effects."
        )
        return interpretation

    @staticmethod
    def plot_shared_gene_count(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        plt.figure(figsize=(10, 6))
        sns.histplot(
            df["num_of_shared_genes_in_path"], bins=30, color="lightgreen", alpha=0.7
        )
        plt.title(
            "Number of Shared Genes Between Drugs and Diseases",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Number of Shared Genes")
        plt.ylabel("Drug Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretation = (
            "This chart shows the distribution of the number of shared genes between drugs and diseases. "
            "Most drugs have few shared genes with diseases, "
            "while some drugs have many shared genes, which may act through regulating related genes."
        )
        return interpretation

    @staticmethod
    def plot_score_vs_degree(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="drug_degree", y="score", data=df, alpha=0.6, s=50)
        plt.title("Drug Degree vs Predicted Score", fontsize=14, fontweight="bold")
        plt.xlabel("Drug Degree")
        plt.ylabel("Predicted Score")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretation = (
            "This chart shows the relationship between the degree of drugs in the knowledge graph and the predicted score. "
            "High-scoring drugs are distributed in different degree intervals, "
            "indicating that the model considers not only network centrality, "
            "but also other biological features."
        )
        return interpretation

    @staticmethod
    def plot_overlap_pathways(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        plt.figure(figsize=(10, 6))
        sns.histplot(df["num_of_shared_pathways"], bins=20, color="orange", alpha=0.7)
        plt.title(
            "Number of Overlapping Pathways Between Drugs and Diseases",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Number of Overlapping Pathways")
        plt.ylabel("Drug Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretation = (
            "This chart shows the distribution of the number of overlapping pathways between drugs and diseases. "
            "Drugs with more overlapping pathways may have better therapeutic effects, "
            "as they can regulate the biological pathways related to the disease."
        )
        return interpretation

    @staticmethod
    def plot_key_genes_distribution(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        plt.figure(figsize=(10, 6))
        sns.histplot(df["num_of_key_genes"], bins=20, color="purple", alpha=0.7)
        plt.title("Number of Key Genes Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Number of Key Genes")
        plt.ylabel("Drug Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretation = (
            "This chart shows the distribution of the number of key genes. "
            "Key genes are genes with high centrality in the PPI network, "
            "and drugs with more key genes may have stronger regulatory capabilities."
        )
        return interpretation

    @staticmethod
    def plot_existing_vs_predicted(
        df: pd.DataFrame, filtered_df: pd.DataFrame, output_path: str
    ) -> str:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 3))
        existing_counts = df["existing"].value_counts()
        labels = ["Predicted Drugs", "Known Drugs"]
        colors = ["lightcoral", "lightblue"]
        total = existing_counts.sum()

        def make_autopct(values):
            def my_autopct(pct):
                count = int(round(pct * total / 100.0))
                return f"{count} ({pct:.1f}%)"

            return my_autopct

        plt.pie(
            existing_counts.values,
            labels=labels,
            autopct=make_autopct(existing_counts.values),
            colors=colors,
            startangle=90,
            textprops={"fontsize": 10},
        )

        plt.title(
            "Ratio of Known Drugs to Predicted Drugs", fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretation = (
            "This chart shows the ratio of known drugs to predicted drugs. "
            "Predicted drugs are new potential therapeutic drugs discovered by the model, "
            "while known drugs are drugs already used to treat the disease, "
            "used to verify the accuracy of the model."
        )
        return interpretation

    def render_image(self, chart: dict) -> str:
        """Generate image content"""
        import json
        import base64

        if chart["file"].endswith(".json"):
            with open(chart["file"], "r") as f:
                json_str = f.read()
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON file {chart['file']}: {e}")

                json_data = json.dumps(data)

                chart_id = (
                    os.path.basename(chart["file"])
                    .replace(".json", "")
                    .replace("-", "_")
                )

                return f"""
                <div id="{chart_id}"></div>
                <script>
                    const {chart_id} = {json_data};
                    Plotly.newPlot('{chart_id}', {chart_id}.data, {chart_id}.layout, {chart_id}.config);
                </script>
                """
        else:
            if self.embed_images and os.path.exists(chart["file"]):
                # 读取图片文件并转换为base64
                with open(chart["file"], "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode("utf-8")

                # 根据文件扩展名确定MIME类型
                file_ext = os.path.splitext(chart["file"])[1].lower()
                mime_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".svg": "image/svg+xml",
                }.get(file_ext, "image/png")

                return f"""
                <img src="data:{mime_type};base64,{img_data}" alt="{chart.get('title', 'Chart')}">
                """
            else:
                # 使用文件路径引用
                return f"""
                <img src="{chart['file']}" alt="{chart.get('title', 'Chart')}">
                """

    def render_interpretation(self, chart: dict) -> str:
        chart_id = (
            os.path.basename(chart["file"])
            .replace(".json", "_text")
            .replace(".png", "_text")
            .replace("-", "_")
        )

        def replace_newlines(text):
            return text.replace("\n", "\\n").replace("'", "\\'")

        return f"""
        <div id="{chart_id}"></div>
        <script>
            document.getElementById('{chart_id}').innerHTML =
            marked.parse('{replace_newlines(chart["interpretation"])}');
        </script>
        """

    def _generate_html_report(
        self, df: pd.DataFrame, filtered_df: pd.DataFrame, charts: dict, title: str
    ) -> str:
        """Generate HTML report content"""

        # Calculate basic statistics
        total_drugs = len(filtered_df)
        avg_score = filtered_df["score"].mean()
        max_score = filtered_df["score"].max()
        min_score = filtered_df["score"].min()
        existing_drugs = filtered_df["existing"].sum()
        predicted_drugs = total_drugs - existing_drugs

        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- <link rel="stylesheet" href="https://cdn.datatables.net/2.3.2/css/dataTables.dataTables.min.css"> -->
    <link href="https://cdn.datatables.net/v/dt/jq-3.7.0/dt-2.3.2/b-3.2.4/b-colvis-3.2.4/cr-2.1.1/cc-1.0.6/fc-5.0.4/r-3.0.5/sc-2.4.3/sb-1.8.3/sp-2.3.3/sl-3.0.1/datatables.min.css" rel="stylesheet" integrity="sha384-Z2GZG5p+oa+BLC7daPCXRdtuIQPMJLutXDhl4AjvL+2t4P9co5z8o+JRw8anAeVR" crossorigin="anonymous">
    <link rel="stylesheet" href="https://cdn.datatables.net/columncontrol/1.0.6/css/columnControl.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/responsive/3.0.5/css/responsive.dataTables.min.css">


    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <!-- <script src="https://cdn.datatables.net/2.3.2/js/dataTables.min.js"></script> -->
    <script src="https://cdn.datatables.net/columncontrol/1.0.6/js/dataTables.columnControl.min.js"></script>
    <script src="https://cdn.datatables.net/responsive/3.0.5/js/dataTables.responsive.min.js"></script>
    
    <script src="https://cdn.datatables.net/v/dt/jq-3.7.0/dt-2.3.2/b-3.2.4/b-colvis-3.2.4/cr-2.1.1/cc-1.0.6/fc-5.0.4/r-3.0.5/sc-2.4.3/sb-1.8.3/sp-2.3.3/sl-3.0.1/datatables.min.js" integrity="sha384-02FG7xFilRgezikGLCY5AgHnuDwEt+i8HU/s2pJjEqkaZr1lCYIm0i/wRN4t4dI3" crossorigin="anonymous"></script>

    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js" type="text/javascript" charset="utf-8"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/lib/marked.umd.js" type="text/javascript"></script>
    <script src="https://cdn.jsdelivr.net/npm/clipboard@2.0.11/dist/clipboard.min.js"></script>
    <title>{title}</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 0; 
            background-color: #f5f5f5; 
            display: flex;
        }}
        
        /* Sidebar Styles */
        .sidebar {{
            width: 280px;
            height: 100vh;
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
            color: white;
            position: fixed;
            left: 0;
            top: 0;
            overflow-y: auto;
            transition: transform 0.3s ease;
            z-index: 1000;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }}
        
        .sidebar.collapsed {{
            transform: translateX(-280px);
        }}
        
        .sidebar-header {{
            padding: 20px;
            border-bottom: 1px solid #34495e;
            text-align: center;
        }}
        
        .sidebar-header h3 {{
            margin: 0;
            color: #ecf0f1;
            font-size: 18px;
        }}
        
        .sidebar-toggle {{
            position: absolute;
            left: 300px;
            top: 20px;
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            z-index: 1001;
            transition: background 0.3s ease;
        }}
        
        .sidebar-toggle.collapsed {{
            left: 20px;
        }}
        
        .sidebar-toggle:hover {{
            background: #2980b9;
        }}
        
        .sidebar-toggle.collapsed {{
            left: 20px;
        }}
        
        .nav-menu {{
            padding: 20px 0;
        }}
        
        .nav-item {{
            padding: 0;
            margin: 0;
        }}
        
        .nav-link {{
            display: block;
            padding: 12px 20px;
            color: #bdc3c7;
            text-decoration: none;
            transition: all 0.3s ease;
            border-left: 3px solid transparent;
        }}
        
        .nav-link:hover {{
            background-color: #34495e;
            color: #ecf0f1;
            border-left-color: #3498db;
        }}
        
        .nav-link.active {{
            background-color: #3498db;
            color: white;
            border-left-color: #2980b9;
        }}
        
        .nav-submenu {{
            padding-left: 20px;
            background-color: #2c3e50;
        }}
        
        .nav-submenu .nav-link {{
            padding: 8px 20px;
            font-size: 14px;
        }}
        
        /* Main Content Styles */
        .main-content {{
            flex: 1;
            margin-left: 280px;
            transition: margin-left 0.3s ease;
            min-height: 100vh;
        }}
        
        .main-content.expanded {{
            margin-left: 0;
        }}
        
        .container {{ 
            width: calc(100vw - 320px);
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        
        .expanded > .container {{
            width: calc(100vw - 40px);
        }}
        
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-top: 0; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #34495e; margin-top: 25px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .chart-section {{ margin: 30px 0; }}
        .prompt-section {{ margin: 30px 0; }}
        .prompt-container {{ 
            background-color: #f8f9fa; 
            border-radius: 8px; 
            margin: 10px 0; 
            position: relative;
        }}
        .prompt-container pre {{ 
            text-wrap: auto; 
            margin: 0;
            padding: 15px;
            background-color: #f1f3f4;
            border-radius: 5px;
            border: 1px solid #e0e0e0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.5;
            overflow-x: auto;
        }}
        .copy-button {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.3s ease;
            z-index: 10;
        }}
        .copy-button:hover {{
            background: #2980b9;
        }}
        .copy-button.copied {{
            background: #27ae60;
        }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; max-height: 900px; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .plot-container {{ display: flex; justify-content: center; align-items: center; }}
        .interpretation {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #3498db; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .top-drugs {{ margin: 20px 0; }}
        .dt-column-title {{ color: #000; }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .sidebar {{
                width: 250px;
            }}
            .sidebar.collapsed {{
                transform: translateX(-250px);
            }}
            .main-content {{
                margin-left: 0;
            }}
            .container {{
                width: 95%;
                margin: 10px;
            }}
        }}
    </style>
</head>
<body>
    <!-- Sidebar Toggle Button -->
    <button class="sidebar-toggle" id="sidebarToggle">
        ☰ Navigation
    </button>
    
    <!-- Sidebar Navigation -->
    <div class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <h3>📋 BioMedGPS Explainer</h3>
        </div>
        <nav class="nav-menu">
            <div class="nav-item">
                <a href="#overview" class="nav-link">📊 Analysis Overview</a>
            </div>
            <div class="nav-item">
                <a href="#candidate-drugs" class="nav-link">🏆 Candidate Drugs</a>
            </div>
            <div class="nav-item">
                <a href="#prompt-section" class="nav-link">🤖 ChatGPT Prompt</a>
            </div>
            <div class="nav-item">
                <a href="#visualization" class="nav-link">📈 Visualization Analysis</a>
                <div class="nav-submenu">
                    <a href="#score-distribution" class="nav-link">• Score Distribution</a>
                    <a href="#predicted-score-boxplot" class="nav-link">• Score Boxplot</a>
                    <a href="#disease-similarity-heatmap" class="nav-link">• Disease Similarity</a>
                    <a href="#network-centrality" class="nav-link">• Network Centrality</a>
                    <a href="#shared-genes-pathways" class="nav-link">• Shared Genes & Pathways</a>
                    <a href="#drug-disease-network" class="nav-link">• Drug Network</a>
                    <a href="#shared-gene-count" class="nav-link">• Shared Gene Count</a>
                    <a href="#score-vs-degree" class="nav-link">• Score vs Degree</a>
                    <a href="#shared-gene-count-vs-score" class="nav-link">• Gene Count vs Score</a>
                    <a href="#shared-pathways" class="nav-link">• Shared Pathways</a>
                    <a href="#key-genes-distribution" class="nav-link">• Key Genes Distribution</a>
                    <a href="#existing-vs-predicted" class="nav-link">• Existing vs Predicted</a>
                </div>
            </div>
            <div class="nav-item">
                <a href="#conclusion" class="nav-link">💡 Analysis Conclusion</a>
            </div>
        </nav>
    </div>
    
    <!-- Main Content -->
    <div class="main-content" id="mainContent">
        <div class="container">
            <h1>{title}</h1>
            
            <h2 id="overview">📊 Analysis Overview</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{total_drugs}</div>
                    <div class="stat-label">Total Candidate Drugs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{avg_score:.3f}</div>
                    <div class="stat-label">Average Predicted Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{max_score:.3f}</div>
                    <div class="stat-label">Highest Predicted Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{existing_drugs}</div>
                    <div class="stat-label">Known Drugs</div>
                </div>
            </div>
            
            <h2 id="candidate-drugs">🏆 Top {len(filtered_df)} Candidate Drugs</h2>
            <div class="top-drugs">
                {self.render_table(filtered_df, "top-drugs")}
            </div>
        """

        html_content += """
        <script>
        $(document).ready(function () {
            // DataTable initialization
            $('#top-drugs').DataTable({
                columnControl: [
                    'order',
                    ['searchList', 'search', 'spacer', 'orderAsc', 'orderDesc', 'orderClear']
                ],
                ordering: {
                    indicators: false
                },
                fixedColumns: {
                  start: 2
                },
                layout: {
                    topStart: {
                      buttons: ['colvis']
                    }
                },
                responsive: false,
                scrollX: true,
                scrollY: 300,
                scrollCollapse: false,
                paging: false,
                info: true,
                searching: true,
            });

            // Sidebar toggle functionality
            const sidebar = document.getElementById('sidebar');
            const mainContent = document.getElementById('mainContent');
            const sidebarToggle = document.getElementById('sidebarToggle');
            let sidebarCollapsed = false;

            sidebarToggle.addEventListener('click', function() {
                sidebarCollapsed = !sidebarCollapsed;
                
                if (sidebarCollapsed) {
                    sidebar.classList.add('collapsed');
                    mainContent.classList.add('expanded');
                    sidebarToggle.classList.add('collapsed');
                } else {
                    sidebar.classList.remove('collapsed');
                    mainContent.classList.remove('expanded');
                    sidebarToggle.classList.remove('collapsed');
                }
            });

            // Smooth scrolling for navigation links
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetId = this.getAttribute('href').substring(1);
                    const targetElement = document.getElementById(targetId);
                    
                    if (targetElement) {
                        targetElement.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                        
                        // Update active state
                        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                        this.classList.add('active');
                    }
                });
            });

            // Highlight active section on scroll
            window.addEventListener('scroll', function() {
                const sections = document.querySelectorAll('h2[id], h3[id]');
                const navLinks = document.querySelectorAll('.nav-link');
                
                let current = '';
                sections.forEach(section => {
                    const sectionTop = section.offsetTop;
                    const sectionHeight = section.clientHeight;
                    if (window.pageYOffset >= sectionTop - 200) {
                        current = section.getAttribute('id');
                    }
                });

                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === '#' + current) {
                        link.classList.add('active');
                    }
                });
            });

            // Auto-collapse sidebar on mobile
            function checkMobile() {
                if (window.innerWidth <= 768) {
                    sidebar.classList.add('collapsed');
                    mainContent.classList.add('expanded');
                    sidebarCollapsed = true;
                }
            }
            
            checkMobile();
            window.addEventListener('resize', checkMobile);
            
            new ClipboardJS('.copy-button').on('success', function(e) {
                $('#copy-prompt').text('Copied');
                $('#copy-prompt').addClass('copied');
                setTimeout(function() {
                    $('#copy-prompt').text('Copy');
                    $('#copy-prompt').removeClass('copied');
                }, 3000);
            });
        });
        </script>
        """

        prompt_item = charts.get("prompt", None)
        if prompt_item is not None:
            if os.path.exists(prompt_item["file"]):
                with open(prompt_item["file"], "r") as f:
                    prompt = f.read()
            else:
                prompt = None

            if prompt is not None:
                prompt_safe = prompt.replace('"', '\\"')
                html_content += f"""
                <div class="prompt-section" id="prompt-section">
                    <h3>
                        Prompt for ChatGPT Deep Research
                        <a href="https://chatgpt.com/" target="_blank">
                            <button class="btn btn-primary">Open in ChatGPT</button>
                        </a>
                    </h3>
                    <div class="prompt-container">
                        <button class="copy-button" id="copy-prompt" data-clipboard-text="{prompt_safe}">Copy</button>
                        <pre id="prompt-content">{prompt}</pre>
                    </div>
                    <div class="interpretation">
                        <strong>Interpretation:</strong>{self.render_interpretation(prompt_item)}
                    </div>
                </div>
                """

        html_content += """
        <h2 id="visualization">📈 Visualization Analysis</h2>
        """

        # Add charts
        for chart_type, chart_info in charts.items():
            chart_title = self.get_chart_title(chart_type)
            chart_id = chart_type.replace("_", "-")

            html_content += f"""
            <div class="chart-section" id="{chart_id}">
                <h3>{chart_title}</h3>
                <div class="chart-container">
                    {self.render_image(chart_info)}
                </div>
                <div class="interpretation">
                    <strong>Interpretation:</strong>{self.render_interpretation(chart_info)}
                </div>
            </div>
            """

        html_content += (
            """
        <h2 id="conclusion">💡 Analysis Conclusion</h2>
        <div class="interpretation">
            <p><strong>Main Findings:</strong></p>
            <ul>
                <li>The model successfully identified multiple candidate drugs, including known therapeutic drugs, verifying the effectiveness of the method.</li>
                <li>High-scoring drugs have higher network centrality and shared gene numbers, which is consistent with biological principles.</li>
                <li>Predicted drugs are similar to known drugs in multiple feature dimensions, supporting their potential therapeutic value.</li>
            </ul>
            <p><strong>Suggestions:</strong></p>
            <ul>
                <li>Prioritize the top 10 high-scoring drugs for experimental validation.</li>
                <li>Focus on drugs with high shared gene numbers and overlapping pathways.</li>
                <li>Combine network centrality analysis to select drugs with strong regulatory capabilities.</li>
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 0.9em;">
            <p>Report Generation Time: """
            + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
            <p>Powered by drugs4disease</p>
        </div>
        </div>
    </div>
</body>
</html>
        """
        )

        return html_content
