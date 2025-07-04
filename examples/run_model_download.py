#!/usr/bin/env python3
"""
示例：使用Model类下载和转换wandb模型
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from drugs4disease.model import Model


def main():
    """主函数：演示Model类的使用"""

    # 1. 创建Model实例
    print("=== 创建Model实例 ===")
    model = Model("biomedgps-kge-v1")

    # 2. 获取所有模型信息
    print("\n=== 获取所有模型信息 ===")
    try:
        models = model.load_models()
        print(f"找到 {len(models)} 个模型")

        # 显示前几个模型的基本信息
        for i, model_info in enumerate(models[:3]):
            print(f"\n模型 {i+1}:")
            print(f"  Run ID: {model_info['run_id']}")
            print(f"  Run Name: {model_info['run_name']}")
            print(f"  State: {model_info['state']}")
            print(f"  Artifacts数量: {len(model_info['artifacts'])}")

            # 显示artifacts信息
            for artifact in model_info['artifacts'][:2]:  # 只显示前2个
                print(f"    - {artifact['name']} (类型: {artifact['type']})")

    except Exception as e:
        print(f"获取模型信息失败: {e}")
        return

    # 3. 下载并转换特定模型
    print("\n=== 下载并转换模型 ===")
    run_id = "6vlvgvfq"  # 您提供的示例run_id

    try:
        print(f"开始下载和转换模型 {run_id}...")
        converted_files = model.download_and_convert(run_id)

        print("转换完成！生成的文件：")
        for file_type, file_path in converted_files.items():
            print(f"  {file_type}: {file_path}")

    except Exception as e:
        print(f"下载和转换失败: {e}")


if __name__ == "__main__":
    main() 
