import unittest
import pandas as pd
import os
from drugs4disease.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'score': [0.6, 0.4, 0.8, 0.9, 0.3],
            'shared_gene_counts': [2, 0, 5, 8, 1],
            'drug_degree': [10, 5, 20, 15, 8],
            'overlap_pathways_count': [1, 0, 3, 5, 0],
            'num_key_genes': [2, 0, 4, 6, 1],
            'existing': [False, True, False, True, False]
        })
        self.outdir = 'test_figs'
        os.makedirs(self.outdir, exist_ok=True)

    def test_plot_score_distribution(self):
        out = os.path.join(self.outdir, 'score.png')
        text = Visualizer.plot_score_distribution(self.df, out)
        self.assertTrue(os.path.exists(out))
        self.assertIn('得分分布', text)

    def test_plot_shared_gene_count(self):
        out = os.path.join(self.outdir, 'gene.png')
        text = Visualizer.plot_shared_gene_count(self.df, out)
        self.assertTrue(os.path.exists(out))
        self.assertIn('共享基因', text)

    def test_plot_score_vs_degree(self):
        out = os.path.join(self.outdir, 'scatter.png')
        text = Visualizer.plot_score_vs_degree(self.df, out)
        self.assertTrue(os.path.exists(out))
        self.assertIn('连接度', text)

    def test_plot_overlap_pathways(self):
        out = os.path.join(self.outdir, 'pathways.png')
        text = Visualizer.plot_overlap_pathways(self.df, out)
        self.assertTrue(os.path.exists(out))
        self.assertIn('重叠通路', text)

    def test_plot_key_genes_distribution(self):
        out = os.path.join(self.outdir, 'key_genes.png')
        text = Visualizer.plot_key_genes_distribution(self.df, out)
        self.assertTrue(os.path.exists(out))
        self.assertIn('关键基因', text)

    def test_plot_existing_vs_predicted(self):
        out = os.path.join(self.outdir, 'existing.png')
        text = Visualizer.plot_existing_vs_predicted(self.df, out)
        self.assertTrue(os.path.exists(out))
        self.assertIn('已知药物', text)

    def tearDown(self):
        # 清理测试文件
        import shutil
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

if __name__ == '__main__':
    unittest.main() 