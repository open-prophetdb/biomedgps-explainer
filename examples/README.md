# 示例脚本

本目录包含drugs4disease包的使用示例脚本。

## 脚本说明

### run_data_validation.py
**用途**: 验证BioMedGPS数据文件的存在和格式，支持自动解压缩ZIP文件
**依赖**: 无额外依赖，仅使用Python标准库
**运行**: `python3 examples/run_data_validation.py`

功能：
- 自动检测并解压缩ZIP格式的模型文件
- 验证以下文件：
  - `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/annotated_entities.tsv`
  - `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/knowledge_graph.tsv`
  - `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/entity_embeddings.tsv`
  - `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/relation_type_embeddings.tsv`

### run_data_statistics.py
**用途**: 展示BioMedGPS数据文件的基本信息和统计，支持自动解压缩ZIP文件
**依赖**: 无额外依赖，仅使用Python标准库
**运行**: `python3 examples/run_data_statistics.py`

功能：
- 自动检测并解压缩ZIP格式的模型文件
- 统计实体类型分布
- 统计关系类型分布
- 显示嵌入向量维度
- 提供数据文件概览

### run_full_example.py
**用途**: 完整的药物发现分析工作流程演示，支持自动解压缩ZIP文件
**依赖**: 需要安装drugs4disease包及其依赖
**运行**: `python3 examples/run_full_example.py`

完整流程：
1. 自动检测并解压缩ZIP格式的模型文件
2. 运行药物预测分析
3. 应用过滤条件
4. 生成可视化报告
5. 输出综合分析报告

## 自动解压缩功能

所有脚本都支持自动检测和解压缩ZIP格式的模型文件：

### 支持的ZIP文件格式
脚本会自动检测模型目录下的以下ZIP文件：
- `annotated_entities.tsv.zip` - 实体注释信息
- `knowledge_graph.tsv.zip` - 知识图谱三元组
- `entity_embeddings.tsv.zip` - 实体嵌入向量
- `relation_type_embeddings.tsv.zip` - 关系类型嵌入向量

### 解压缩逻辑
1. 检查模型目录是否存在
2. 对于每个必需的TSV文件，检查是否存在对应的ZIP文件
3. 如果TSV文件不存在但ZIP文件存在，则自动解压缩
4. 解压缩完成后继续正常流程

### 使用场景
- **ZIP格式**: 将模型文件以ZIP格式放在模型目录下，脚本会自动解压缩
- **已解压缩**: 如果TSV文件已存在，脚本会直接使用，跳过解压缩
- **混合格式**: 支持部分文件为ZIP格式，部分文件已解压缩

## 使用顺序

建议按以下顺序运行示例：

1. **首先验证数据文件**:
   ```bash
   python3 examples/run_data_validation.py
   ```

2. **查看数据统计**:
   ```bash
   python3 examples/run_data_statistics.py
   ```

3. **安装依赖**:
   ```bash
   pip install -e .
   ```

4. **运行完整示例**:
   ```bash
   python3 examples/run_full_example.py
   ```

## 数据文件准备

### 方法1：使用ZIP压缩文件（推荐）
1. 将BioMedGPS模型ZIP文件放在模型目录下：
   ```
   data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/
   ├── annotated_entities.tsv.zip
   ├── knowledge_graph.tsv.zip
   ├── entity_embeddings.tsv.zip
   └── relation_type_embeddings.tsv.zip
   ```
2. 运行示例脚本，会自动解压缩

### 方法2：使用解压缩后的TSV文件
1. 手动解压缩模型文件到`data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/`目录
2. 确保包含以下文件：
   - `annotated_entities.tsv`
   - `knowledge_graph.tsv`
   - `entity_embeddings.tsv`
   - `relation_type_embeddings.tsv`

## 输出文件

运行完整示例后，会在项目根目录的`results/`文件夹中生成：
- `annotated_drugs.xlsx`: 完整注释的药物列表
- `filtered_drugs.xlsx`: 过滤后的药物列表
- `visualization_report/`: 可视化图表和报告

## 注意事项

- 所有脚本都会自动从examples目录向上查找项目根目录
- 数据文件路径会自动调整为相对于项目根目录的路径
- 输出文件会保存在项目根目录的results文件夹中
- 支持自动解压缩ZIP格式的模型文件
- 如果数据文件不存在，脚本会提供清晰的错误信息和使用指导 