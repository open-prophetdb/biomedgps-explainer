# 示例脚本

本目录包含drugs4disease包的使用示例脚本。

## 脚本说明

### test_data_files.py
**用途**: 验证BioMedGPS数据文件的存在和格式
**依赖**: 无额外依赖，仅使用Python标准库
**运行**: `python3 examples/test_data_files.py`

验证以下文件：
- `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/annotated_entities.tsv`
- `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/knowledge_graph.tsv`
- `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/entity_embeddings.tsv`
- `data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/relation_type_embeddings.tsv`

### simple_example.py
**用途**: 展示BioMedGPS数据文件的基本信息和统计
**依赖**: 无额外依赖，仅使用Python标准库
**运行**: `python3 examples/simple_example.py`

功能：
- 统计实体类型分布
- 统计关系类型分布
- 显示嵌入向量维度
- 提供数据文件概览

### example_usage.py
**用途**: 完整的药物发现分析工作流程演示
**依赖**: 需要安装drugs4disease包及其依赖
**运行**: `python3 examples/example_usage.py`

完整流程：
1. 运行药物预测分析
2. 应用过滤条件
3. 生成可视化报告
4. 输出综合分析报告

## 使用顺序

建议按以下顺序运行示例：

1. **首先验证数据文件**:
   ```bash
   python3 examples/test_data_files.py
   ```

2. **查看数据统计**:
   ```bash
   python3 examples/simple_example.py
   ```

3. **安装依赖**:
   ```bash
   pip install -e .
   ```

4. **运行完整示例**:
   ```bash
   python3 examples/example_usage.py
   ```

## 输出文件

运行完整示例后，会在项目根目录的`results/`文件夹中生成：
- `annotated_drugs.xlsx`: 完整注释的药物列表
- `filtered_drugs.xlsx`: 过滤后的药物列表
- `visualization_report/`: 可视化图表和报告

## 注意事项

- 所有脚本都会自动从examples目录向上查找项目根目录
- 数据文件路径会自动调整为相对于项目根目录的路径
- 输出文件会保存在项目根目录的results文件夹中
- 如果数据文件不存在，脚本会提供清晰的错误信息 