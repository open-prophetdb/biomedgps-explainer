# BioMedGPS Explainer

基于知识图谱的潜在药物发现与可视化工具包。

## 功能特性

- **药物预测**: 基于知识图谱嵌入(KGE)模型预测潜在药物
- **网络分析**: 药物-疾病-基因路径分析、中心性计算、PPI网络分析
- **通路富集**: 药物与疾病的通路重叠分析
- **共享注释**: 统计药物与疾病的共享基因和疾病
- **智能筛选**: 支持复杂逻辑表达式的药物筛选
- **可视化**: 自动生成图表和解读报告

## 安装方法

```bash
pip install .
```

## 数据准备

本工具使用BioMedGPS v2知识图谱嵌入模型。支持两种数据文件准备方式：

### 方法1：使用ZIP压缩文件（推荐）
将BioMedGPS模型ZIP文件放在模型目录下，脚本会自动检测并解压缩：

```
data/
└── biomedgps_v2_20250318_TransE_l2_KMkgBhIV/
    ├── annotated_entities.tsv.zip      # 实体注释信息（ZIP格式）
    ├── entity_embeddings.tsv.zip       # 实体嵌入向量（ZIP格式）
    ├── knowledge_graph.tsv.zip         # 知识图谱三元组（ZIP格式）
    └── relation_type_embeddings.tsv.zip # 关系类型嵌入向量（ZIP格式）
```

### 方法2：使用解压缩后的TSV文件
手动解压缩模型文件到以下位置：

```
data/
└── biomedgps_v2_20250318_TransE_l2_KMkgBhIV/
    ├── annotated_entities.tsv      # 实体注释信息
    ├── entity_embeddings.tsv       # 实体嵌入向量
    ├── knowledge_graph.tsv         # 知识图谱三元组
    └── relation_type_embeddings.tsv # 关系类型嵌入向量
```

**自动解压缩功能**：脚本会自动检测模型目录下的ZIP文件（如 `annotated_entities.tsv.zip`）并解压缩为对应的TSV文件。如果TSV文件已存在，则跳过解压缩步骤。

详细的数据文件格式说明请参考 `data/README.md`。

## 示例脚本使用

### 1. 验证数据文件
```bash
python3 examples/run_data_validation.py
```

### 2. 查看数据统计
```bash
python3 examples/run_data_statistics.py
```

### 3. 安装依赖并运行完整示例
```bash
pip install -e .
python3 examples/run_full_example.py
```

## CLI 用法示例

### 1. 一键完成所有分析
```bash
drugs4disease run --disease "MESH:D001249" \
  --entity-file ./data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/annotated_entities.tsv \
  --knowledge-graph ./data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/knowledge_graph.tsv \
  --entity-embeddings ./data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/entity_embeddings.tsv \
  --relation-embeddings ./data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/relation_type_embeddings.tsv \
  --output-dir ./results \
  --model TransE_l2 \
  --top-n-diseases 50 \
  --gamma 12.0 \
  --threshold 0.5 \
  --relation-type "DGIDB::OTHER::Gene:Compound"
```

### 2. 筛选药物列表
```bash
drugs4disease filter --expression 'score >= 0.5 && num_of_shared_genes_in_path > 0 && existing == False' \
  --input ./results/annotated_drugs.xlsx \
  --output ./results/final_drugs.xlsx
```

### 3. 生成图表与解读
```bash
# 生成所有类型的可视化
drugs4disease visualize --input ./results/annotated_drugs.xlsx --output-dir ./results/figures

# 生成特定类型的可视化
drugs4disease visualize --input ./results/annotated_drugs.xlsx --output-dir ./results/figures --viz-type score_distribution
```

## 参数说明

### run 命令参数
- `--disease`: 疾病名称或ID (必需)
- `--entity-file`: 实体信息文件 (必需)
- `--knowledge-graph`: 知识图谱文件 (必需)
- `--entity-embeddings`: 实体嵌入文件 (必需)
- `--relation-embeddings`: 关系嵌入文件 (必需)
- `--output-dir`: 输出目录 (必需)
- `--model`: KGE模型类型 (默认: TransE_l2)
- `--top-n-diseases`: 相似疾病数量 (默认: 100)
- `--gamma`: Gamma参数 (默认: 12.0)
- `--threshold`: 药物筛选阈值 (默认: 0.5)
- `--relation-type`: 关系类型 (默认: DGIDB::OTHER::Gene:Compound)

### filter 命令参数
- `--expression`: 筛选表达式，支持 `&&`、`||`、`==`、`!=`、`>=`、`<=` 等
- `--input`: 输入文件 annotated_drugs.xlsx
- `--output`: 输出文件 final_drugs.xlsx

### visualize 命令参数
- `--input`: 输入文件 annotated_drugs.xlsx
- `--output-dir`: 图表输出目录
- `--viz-type`: 可视化类型 (默认: all，可选: score_distribution, top_drugs_bar, disease_similarity_heatmap, network_centrality, shared_genes_pathways, drug_disease_network)

## 输出文件结构

```
results/
  annotated_drugs.xlsx          # 完整注释的药物列表
  predicted_drugs.xlsx          # 初步预测结果
  shared_genes_pathways.xlsx    # 共享基因和通路
  shared_diseases.xlsx          # 共享疾病
  network_annotations.xlsx      # 网络分析结果
  filtered_drugs.xlsx           # 筛选后药物
  figures/
    score_distribution.png      # 得分分布图
    top_drugs_bar.png           # 前N药物柱状图
    disease_similarity_heatmap.png # 疾病相似性热图
    network_centrality.png      # 网络中心性分析
    shared_genes_pathways.png   # 共享基因通路分析
    drug_disease_network.png    # 药物-疾病网络图
    analysis_report.html        # 综合分析报告
```

## 输出字段说明

`annotated_drugs.xlsx` 包含以下字段：
- `drug_id`: 药物ID
- `drug_name`: 药物名称
- `score`: KGE预测得分
- `num_of_shared_genes_in_path`: 与疾病共享的基因数量
- `paths`: 药物-疾病两跳路径
- `existing`: 是否为已知治疗药物
- `num_of_shared_pathways`: 重叠通路数量
- `shared_pathways`: 重叠通路名称
- `key_genes`: 关键基因
- `num_of_key_genes`: 关键基因数量
- `drug_degree`: 药物在知识图谱中的连接度
- `num_of_shared_genes`: 共享基因数量（详细统计）
- `shared_gene_names`: 共享基因名称
- `num_of_shared_diseases`: 共享疾病数量
- `shared_disease_names`: 共享疾病名称

## 筛选表达式示例

```bash
# 高分且非现有药物
'score >= 0.7 && existing == False'

# 有共享基因且重叠通路
'num_of_shared_genes_in_path > 0 && num_of_shared_pathways > 0'

# 复杂条件组合
'score >= 0.5 && num_of_shared_genes_in_path > 2 && drug_degree >= 10 && existing == False'

# 多条件或关系
'score >= 0.8 || (num_of_shared_genes_in_path > 5 && num_of_shared_pathways > 3)'
```

## 单元测试

```bash
python -m unittest discover tests
```

## 依赖

- Python 3.8+
- click>=8.0.0
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- networkx>=2.6.0
- gseapy>=0.12.0
- gprofiler-official>=1.0.0
- torch>=1.9.0
- openpyxl>=3.0.0
- requests>=2.25.0
- scikit-learn>=1.0.0

## 贡献

欢迎提交 issue 或 PR 以完善功能。

## 许可证

MIT License 