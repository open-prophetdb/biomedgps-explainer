# 数据文件说明

本目录包含BioMedGPS知识图谱嵌入模型的相关文件，用于药物发现分析。

## 文件结构

```
data/
└── biomedgps_v2_20250318_TransE_l2_KMkgBhIV/
    ├── annotated_entities.tsv      # 实体注释信息
    ├── entity_embeddings.tsv       # 实体嵌入向量
    ├── knowledge_graph.tsv         # 知识图谱三元组
    └── relation_type_embeddings.tsv # 关系类型嵌入向量
```

## 文件格式说明

### annotated_entities.tsv
包含所有实体的注释信息，字段包括：
- `id`: 实体ID
- `name`: 实体名称
- `label`: 实体类型标签
- `resource`: 数据来源
- `description`: 实体描述
- `synonyms`: 同义词
- `pmids`: 相关PubMed ID
- `taxid`: 分类ID
- `xrefs`: 交叉引用

### entity_embeddings.tsv
包含所有实体的嵌入向量，字段包括：
- `embedding_id`: 嵌入向量ID
- `entity_id`: 实体ID
- `entity_type`: 实体类型
- `entity_name`: 实体名称
- `embedding`: 嵌入向量（以|分隔的数值）

### knowledge_graph.tsv
包含知识图谱的三元组关系，字段包括：
- `relation_type`: 关系类型
- `resource`: 数据来源
- `pmids`: 相关PubMed ID
- `key_sentence`: 关键句子
- `source_id`: 源实体ID
- `source_type`: 源实体类型
- `target_id`: 目标实体ID
- `target_type`: 目标实体类型
- `source_name`: 源实体名称
- `target_name`: 目标实体名称

### relation_type_embeddings.tsv
包含关系类型的嵌入向量，格式与entity_embeddings.tsv类似。

## 使用说明

这些文件是BioMedGPS v2模型的核心数据，支持以下功能：
- 药物-疾病关联预测
- 基因-药物相互作用分析
- 生物医学实体相似性计算
- 知识图谱路径分析

## 注意事项

- 文件较大，建议使用流式读取
- 嵌入向量为高维数值，需要适当的内存管理
- 实体ID遵循标准生物医学命名规范（如ENTREZ、MESH等）

本目录仅作结构说明，实际数据需用户自备。 