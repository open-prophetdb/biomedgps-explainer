# Model类使用指南

## 概述

`Model`类是一个用于从Weights & Biases (wandb) 获取模型信息和下载转换模型文件的工具类。它提供了以下主要功能：

1. 获取指定项目下的所有模型信息，包括Artifacts
2. 下载指定模型的所有相关文件
3. 将模型文件转换成本项目需要的格式

## 安装依赖

确保已安装以下Python包：

```bash
pip install wandb numpy pandas
```

## 基本使用

### 1. 创建Model实例

```python
from drugs4disease.model import Model

# 创建Model实例，指定wandb项目名称
model = Model("biomedgps-kge-v1")
```

### 2. 获取所有模型信息

```python
# 获取项目下的所有模型信息
models = model.get_all_models()

# 查看模型信息
for model_info in models:
    print(f"Run ID: {model_info['run_id']}")
    print(f"Run Name: {model_info['run_name']}")
    print(f"State: {model_info['state']}")
    print(f"Artifacts数量: {len(model_info['artifacts'])}")
```

返回的模型信息包含：
- `run_id`: 运行ID
- `run_name`: 运行名称
- `state`: 运行状态
- `created_at`: 创建时间
- `updated_at`: 更新时间
- `config`: 配置信息
- `summary`: 摘要信息
- `artifacts`: 所有artifacts的详细信息

### 3. 下载模型文件

```python
# 下载指定模型的所有文件
run_id = "sdihyrpu"
output_dir = "./downloaded_models"
model_dir = model.download_model(run_id, output_dir)
print(f"模型文件下载到: {model_dir}")
```

### 4. 转换模型文件

```python
# 转换模型文件成本项目需要的格式
converted_files = model.convert_model_files(model_dir, "./converted_models")
print(f"转换后的文件: {converted_files}")
```

### 5. 一键下载并转换

```python
# 下载并转换的完整流程
converted_files = model.download_and_convert(run_id, "./final_models")
print(f"转换完成，文件位置: {converted_files}")
```

## 输出文件格式

转换后的文件包括：

### 1. entity_embeddings.tsv
实体嵌入文件，包含以下列：
- `embedding_id`: 嵌入ID (格式: label::id)
- `entity_id`: 实体ID
- `entity_type`: 实体类型
- `entity_name`: 实体名称
- `embedding`: 嵌入向量 (用"|"分隔的数值)

### 2. relation_type_embeddings.tsv
关系类型嵌入文件，包含以下列：
- `id`: 关系类型ID
- `embedding`: 嵌入向量 (用"|"分隔的数值)

### 3. unfound_records.tsv (可选)
未找到对应元数据的记录文件。

## 错误处理

Model类包含完善的错误处理机制：

- 自动验证文件格式
- 详细的日志记录
- 异常信息提示
- 文件完整性检查

## 示例脚本

运行示例脚本：

```bash
cd examples
python run_model_download.py
```

## 注意事项

1. **wandb登录**: 使用前需要确保已正确登录wandb
2. **网络连接**: 下载大文件时需要稳定的网络连接
3. **磁盘空间**: 确保有足够的磁盘空间存储模型文件
4. **文件权限**: 确保对输出目录有写入权限

## 高级用法

### 自定义文件查找

```python
# 自定义文件查找逻辑
def custom_find_file(model_path, pattern):
    # 自定义查找逻辑
    pass

# 可以继承Model类并重写_find_file方法
```

### 批量处理

```python
# 批量下载多个模型
run_ids = ["run1", "run2", "run3"]
for run_id in run_ids:
    try:
        converted_files = model.download_and_convert(run_id, f"./models/{run_id}")
        print(f"成功处理模型 {run_id}")
    except Exception as e:
        print(f"处理模型 {run_id} 失败: {e}")
```

## 故障排除

### 常见问题

1. **wandb登录失败**
   - 检查网络连接
   - 确认wandb API密钥正确
   - 运行 `wandb login` 重新登录

2. **文件下载失败**
   - 检查网络连接
   - 确认模型ID正确
   - 检查磁盘空间

3. **文件转换失败**
   - 检查输入文件格式
   - 确认文件完整性
   - 查看详细错误日志

### 获取帮助

如果遇到问题，可以：

1. 查看日志输出获取详细错误信息
2. 检查输入参数是否正确
3. 确认文件路径和权限
4. 参考示例代码进行调试 