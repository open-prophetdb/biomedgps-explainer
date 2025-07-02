import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Visualizer:
    """
    提供常见图表的生成与解读文字输出。
    """
    @staticmethod
    def plot_score_distribution(df: pd.DataFrame, output_path: str) -> str:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['score'], bins=30, kde=True, color='skyblue', alpha=0.7)
        plt.title('药物预测得分分布', fontsize=14, fontweight='bold')
        plt.xlabel('预测得分', fontsize=12)
        plt.ylabel('药物数量', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        interpretation = (
            '该图展示了候选药物的KGE预测得分分布。大部分药物得分较低，少数药物得分较高。'
            '高分药物（得分>0.7）值得重点关注，可能具有较好的治疗效果。'
        )
        return interpretation

    @staticmethod
    def plot_shared_gene_count(df: pd.DataFrame, output_path: str) -> str:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['shared_gene_counts'], bins=30, color='lightgreen', alpha=0.7)
        plt.title('药物与疾病共享基因数量分布', fontsize=14, fontweight='bold')
        plt.xlabel('共享基因数量', fontsize=12)
        plt.ylabel('药物数量', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        interpretation = (
            '该图展示了药物与疾病共享基因数量的分布。大多数药物与疾病共享基因较少，'
            '部分药物具有较多共享基因，这些药物可能通过调节相关基因发挥作用。'
        )
        return interpretation

    @staticmethod
    def plot_score_vs_degree(df: pd.DataFrame, output_path: str) -> str:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='drug_degree', y='score', data=df, alpha=0.6, s=50)
        plt.title('药物连接度与预测得分关系', fontsize=14, fontweight='bold')
        plt.xlabel('药物连接度', fontsize=12)
        plt.ylabel('预测得分', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        interpretation = (
            '该图展示了药物在知识图谱中的连接度与预测得分的关系。'
            '高分药物分布于不同连接度区间，说明模型不仅依赖于网络中心性，'
            '还考虑了其他生物学特征。'
        )
        return interpretation

    @staticmethod
    def plot_overlap_pathways(df: pd.DataFrame, output_path: str) -> str:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['overlap_pathways_count'], bins=20, color='orange', alpha=0.7)
        plt.title('药物与疾病重叠通路数量分布', fontsize=14, fontweight='bold')
        plt.xlabel('重叠通路数量', fontsize=12)
        plt.ylabel('药物数量', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        interpretation = (
            '该图展示了药物与疾病重叠通路数量的分布。重叠通路数量较多的药物'
            '可能具有更好的治疗效果，因为它们能够调节疾病相关的生物学通路。'
        )
        return interpretation

    @staticmethod
    def plot_key_genes_distribution(df: pd.DataFrame, output_path: str) -> str:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['num_key_genes'], bins=20, color='purple', alpha=0.7)
        plt.title('药物关键基因数量分布', fontsize=14, fontweight='bold')
        plt.xlabel('关键基因数量', fontsize=12)
        plt.ylabel('药物数量', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        interpretation = (
            '该图展示了药物关键基因数量的分布。关键基因是在PPI网络中具有高中心性的基因，'
            '关键基因数量较多的药物可能具有更强的调控能力。'
        )
        return interpretation

    @staticmethod
    def plot_existing_vs_predicted(df: pd.DataFrame, output_path: str) -> str:
        plt.figure(figsize=(8, 6))
        existing_counts = df['existing'].value_counts()
        colors = ['lightcoral', 'lightblue']
        plt.pie(existing_counts.values, labels=['预测药物', '已知药物'], 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('已知药物与预测药物比例', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        interpretation = (
            '该图展示了已知药物与预测药物的比例。预测药物是模型发现的新潜在治疗药物，'
            '而已知药物是已经用于治疗该疾病的药物，用于验证模型的准确性。'
        )
        return interpretation 