# ZIP自动解压缩功能

## 概述

BioMedGPS Explainer包现在支持自动检测和解压缩ZIP格式的模型文件。这个功能让用户可以更方便地使用压缩的模型文件，无需手动解压缩。

## 支持的文件格式

脚本会自动检测模型目录下的以下ZIP文件：

- `annotated_entities.tsv.zip` - 实体注释信息
- `knowledge_graph.tsv.zip` - 知识图谱三元组
- `entity_embeddings.tsv.zip` - 实体嵌入向量
- `relation_type_embeddings.tsv.zip` - 关系类型嵌入向量

## 工作原理

### 解压缩逻辑

1. **检查模型目录**：脚本首先检查模型目录是否存在
2. **逐个文件检查**：对于每个必需的TSV文件，检查是否存在对应的ZIP文件
3. **智能解压缩**：
   - 如果TSV文件不存在但ZIP文件存在 → 自动解压缩
   - 如果TSV文件已存在 → 跳过解压缩
   - 如果TSV文件和ZIP文件都不存在 → 报告错误
4. **继续正常流程**：解压缩完成后继续正常的数据分析流程

### 支持的场景

- **完全ZIP格式**：所有模型文件都是ZIP格式
- **混合格式**：部分文件为ZIP格式，部分文件已解压缩
- **完全解压缩**：所有文件都已解压缩，跳过ZIP处理

## 使用方法

### 1. 准备ZIP文件

将BioMedGPS模型ZIP文件放在模型目录下：

```
data/
└── biomedgps_v2_20250318_TransE_l2_KMkgBhIV/
    ├── annotated_entities.tsv.zip
    ├── knowledge_graph.tsv.zip
    ├── entity_embeddings.tsv.zip
    └── relation_type_embeddings.tsv.zip
```

### 2. 运行脚本

所有示例脚本都支持自动解压缩：

```bash
# 验证数据文件（包含ZIP解压缩）
python3 examples/run_data_validation.py

# 查看数据统计（包含ZIP解压缩）
python3 examples/run_data_statistics.py

# 完整分析流程（包含ZIP解压缩）
python3 examples/run_full_example.py
```

## 实现细节

### 核心函数

```python
def extract_zip_if_needed(zip_path, extract_dir):
    """如果ZIP文件存在，则解压缩到指定目录"""
    if os.path.exists(zip_path):
        print(f"发现ZIP文件: {zip_path}")
        print(f"正在解压缩到: {extract_dir}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✓ 解压缩完成")
            return True
        except Exception as e:
            print(f"✗ 解压缩失败: {e}")
            return False
    return False

def extract_model_files_if_needed(model_dir):
    """检查并解压缩模型目录下的ZIP文件"""
    # 需要解压缩的文件列表
    required_files = [
        "annotated_entities.tsv",
        "knowledge_graph.tsv", 
        "entity_embeddings.tsv",
        "relation_type_embeddings.tsv"
    ]
    
    # 对应的ZIP文件
    zip_files = [
        "annotated_entities.tsv.zip",
        "knowledge_graph.tsv.zip",
        "entity_embeddings.tsv.zip", 
        "relation_type_embeddings.tsv.zip"
    ]
    
    # 逐个检查并解压缩
    for required_file, zip_file in zip(required_files, zip_files):
        file_path = os.path.join(model_dir, required_file)
        zip_path = os.path.join(model_dir, zip_file)
        
        if not os.path.exists(file_path) and os.path.exists(zip_path):
            extract_zip_if_needed(zip_path, model_dir)
```

### 错误处理

- **ZIP文件损坏**：捕获解压缩异常并报告错误
- **权限问题**：处理文件权限相关的错误
- **磁盘空间不足**：处理解压缩过程中的磁盘空间问题

## 优势

1. **用户友好**：无需手动解压缩文件
2. **自动化**：脚本自动处理文件格式检测
3. **灵活性**：支持多种文件格式组合
4. **效率**：避免重复解压缩已存在的文件
5. **错误处理**：提供清晰的错误信息和解决建议

## 测试验证

### 测试场景

1. **完全ZIP格式**：所有文件都是ZIP格式
2. **混合格式**：部分ZIP，部分已解压缩
3. **完全解压缩**：所有文件都已解压缩
4. **错误情况**：ZIP文件损坏、权限问题等

### 验证命令

```bash
# 测试ZIP解压缩功能
python3 examples/run_data_validation.py

# 验证解压缩后的文件
ls -la data/biomedgps_v2_20250318_TransE_l2_KMkgBhIV/

# 运行完整流程验证
python3 examples/run_data_statistics.py
```

## 更新历史

- **2024-07-02**：实现ZIP自动解压缩功能
- 支持模型目录下的ZIP文件检测和解压缩
- 更新所有示例脚本以支持新功能
- 更新文档说明新的使用方法

## 注意事项

1. **磁盘空间**：解压缩后的文件可能很大，确保有足够的磁盘空间
2. **文件权限**：确保脚本有读取ZIP文件和写入解压缩文件的权限
3. **网络下载**：如果ZIP文件很大，建议使用稳定的网络连接下载
4. **备份**：建议保留原始ZIP文件作为备份

## 故障排除

### 常见问题

1. **解压缩失败**
   - 检查ZIP文件是否完整
   - 确认磁盘空间充足
   - 检查文件权限

2. **文件格式错误**
   - 确认ZIP文件包含正确的TSV文件
   - 检查文件名是否匹配预期格式

3. **权限问题**
   - 确保脚本有读取和写入权限
   - 检查目录权限设置

### 调试信息

脚本会输出详细的解压缩过程信息：

```
检查模型目录: /path/to/model/dir
解压缩 annotated_entities.tsv.zip...
发现ZIP文件: /path/to/annotated_entities.tsv.zip
正在解压缩到: /path/to/model/dir
✓ 解压缩完成
✓ 成功解压缩 4 个文件
``` 