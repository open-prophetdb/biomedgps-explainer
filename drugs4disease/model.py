import os
import json
import time
import wandb
import shutil
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model:
    """
    Model类用于从wandb获取模型信息和下载转换模型文件
    """

    def __init__(self, project_name: str):
        """
        初始化Model类
        
        Args:
            project_name (str): wandb项目名称
        """
        self.project_name = project_name
        self.api = None
        self._login()
        self.model_root_dir = os.path.expanduser("~/.biomedgps-explainer/models")
        os.makedirs(self.model_root_dir, exist_ok=True)
        self.model_info_file = os.path.join(self.model_root_dir, "models_info.json")

    def _login(self):
        """登录wandb"""
        try:
            wandb.login()
            self.api = wandb.Api()
            logger.info("Successfully logged in to wandb")
        except Exception as e:
            logger.error(f"Failed to login to wandb: {e}")
            raise

    def get_all_models(self) -> List[Dict[str, Any]]:
        """
        获取指定项目下的所有模型，返回模型的详细信息，包括Artifacts
        
        Returns:
            List[Dict[str, Any]]: 模型信息列表，每个字典包含模型的基本信息和artifacts
        """
        try:
            # 获取项目下的所有运行
            runs = self.api.runs(self.project_name)
            models_info = []

            for run in runs:
                model_info = {
                    'run_id': run.id,
                    'run_name': run.name,
                    'run_title': run.display_name,
                    'state': run.state,
                    'created_at': run.created_at,
                    # 'updated_at': run.updated_at,
                    'config': run.config,
                    'summary': run.summary,
                    'artifacts': []
                }

                # 获取该运行的所有artifacts
                try:
                    for artifact in run.logged_artifacts():
                        artifact_info = {
                            'name': artifact.name,
                            'version': artifact.version,
                            'type': artifact.type,
                            'description': artifact.description,
                            'metadata': artifact.metadata,
                            'size': artifact.size,
                            'created_at': artifact.created_at,
                            'updated_at': artifact.updated_at
                        }
                        model_info['artifacts'].append(artifact_info)
                except Exception as e:
                    logger.warning(f"Failed to get artifacts for run {run.id}: {e}")

                models_info.append(model_info)
                logger.info(f"Retrieved model info for run {run.id}: {run.name}")

            self._save_models_info(models_info)
            logger.info(f"Successfully retrieved {len(models_info)} models from project {self.project_name}")
            return models_info

        except Exception as e:
            logger.error(f"Failed to get models from project {self.project_name}: {e}")
            raise

    def load_models(self) -> List[Dict[str, Any]]:
        """加载模型信息"""
        # Check the creation time of the file, if it is older than 1 day, download the models info again
        if os.path.exists(self.model_info_file):
            created_at = os.path.getctime(self.model_info_file)
            if time.time() - created_at > 86400:
                return self.get_all_models()
            else:
                with open(self.model_info_file, "r") as f:
                    return json.load(f)
        else:
            return self.get_all_models()

    def _save_models_info(self, models_info: List[Dict[str, Any]]) -> None:
        """保存模型信息"""
        def safe_json(obj):
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)

        with open(self.model_info_file, "w") as f:
            json.dump(models_info, f, default=safe_json)

    def download_model(self, run_id: str) -> str:
        """
        指定模型id，下载所有相关文件至指定目录
        
        Args:
            run_id (str): 模型运行ID
            output_dir (str): 输出目录路径
            
        Returns:
            str: 下载后的模型目录路径
        """
        try:
            # 获取运行信息
            run = self.api.run(f"{self.project_name}/{run_id}")
            run_name = run.name

            # 创建输出目录
            model_dir = os.path.join(self.model_root_dir, run_name)
            os.makedirs(model_dir, exist_ok=True)

            if self._check_model_dir(model_dir):
                logger.info(f"Model directory {model_dir} is valid, skipping download")
                return model_dir

            logger.info(f"Downloading model {run_id} ({run_name}) to {model_dir}")

            # 获取所有artifacts
            for artifact in run.logged_artifacts():
                artifact_name = artifact.name
                artifact_version = artifact.version
                artifact_type = artifact.type

                # 规范命名：<artifact_type>_<artifact_name>_v<artifact_version>
                normalized_name = f"{artifact_type}_{artifact_name.replace(':', '_')}"
                artifact_dir = os.path.join(model_dir, normalized_name)

                # 确保目录存在
                os.makedirs(artifact_dir, exist_ok=True)

                # 下载artifact到规范命名的目录
                artifact.download(root=artifact_dir)
                logger.info(f"Downloaded {artifact.name} to {artifact_dir}")

                # 将artifact中的文件移动到根目录
                for root, dirs, files in os.walk(artifact_dir):
                    for file in files:
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(model_dir, file)
                        shutil.move(src_path, dst_path)

                # 删除空目录
                shutil.rmtree(artifact_dir)

            logger.info(f"✅ Successfully downloaded model {run_id} to {model_dir}")
            return model_dir

        except Exception as e:
            logger.error(f"❌ Failed to download model {run_id}: {e}")
            raise

    def _check_model_dir(self, model_dir: str) -> bool:
        """检查模型目录是否存在"""
        if not os.path.exists(model_dir):
            return False

        expected_files = self._get_expected_files(model_dir)
        for file_name, file_path in expected_files.items():
            if file_path is None:
                return False
        return True

    def _get_expected_files(self, model_dir: str) -> Dict[str, str]:
        """获取模型目录下的所有文件"""
        model_path = Path(model_dir)
        entity_emb_fpath = self._find_file(model_path, "_entity.npy")
        entity_metadata_fpath = self._find_file(
            model_path, "annotated_entities.tsv.tar.gz"
        )
        relation_emb_fpath = self._find_file(model_path, "_relation.npy")
        entity_id_map_fpath = self._find_file(model_path, "entities.tsv.tar.gz")
        relation_type_id_map_fpath = self._find_file(model_path, "relations.tsv.tar.gz")
        knowledge_graph_fpath = self._find_file(model_path, "knowledge_graph.tsv.tar.gz")

        return {
            'entity_embeddings': entity_emb_fpath,
            'entity_metadata': entity_metadata_fpath,
            'relation_embeddings': relation_emb_fpath,
            'entity_id_map': entity_id_map_fpath,
            'relation_type_id_map': relation_type_id_map_fpath,
            'knowledge_graph': knowledge_graph_fpath
        }

    def convert_model_files(self, model_dir: str) -> Dict[str, str]:
        """
        转换模型文件成本项目需要的格式
        
        Args:
            model_dir (str): 模型文件目录
            output_dir (str): 输出目录
            
        Returns:
            Dict[str, str]: 转换后的文件路径字典
        """
        try:
            model_path = Path(model_dir)
            output_path = model_path / "converted_model"
            output_path.mkdir(parents=True, exist_ok=True)

            # 查找模型文件
            expected_files = self._get_expected_files(model_path)

            missing_files = [name for name, path in expected_files.items() if path is None]
            if missing_files:
                raise ValueError(f"Missing required files: {missing_files}")

            logger.info("Starting model file conversion...")

            # 转换实体嵌入
            entity_embedding_fpath = self._convert_entity_embeddings(
                expected_files['entity_embeddings'], expected_files['entity_metadata'], expected_files['entity_id_map'], output_path
            )

            # 转换关系嵌入
            relation_embedding_fpath = self._convert_relation_embeddings(
                expected_files['relation_embeddings'], expected_files['relation_type_id_map'], output_path
            )

            annotated_entities_fpath = self._extract_tar_gz(expected_files['entity_metadata'], output_path)
            knowledge_graph_fpath = self._extract_tar_gz(expected_files['knowledge_graph'], output_path)

            converted_files = {
                'entity_embeddings': str(entity_embedding_fpath),
                'relation_embeddings': str(relation_embedding_fpath),
                'annotated_entities': str(annotated_entities_fpath),
                'knowledge_graph': str(knowledge_graph_fpath),
                'model_dir': str(model_path)
            }

            logger.info(f"Successfully converted model files to {output_path}")
            return converted_files

        except Exception as e:
            logger.error(f"Failed to convert model files: {e}")
            raise

    def _find_file(self, model_path: Path, file_pattern: str) -> Optional[Path]:
        """查找模型文件"""
        for file_path in model_path.rglob("*"):
            if file_pattern.lower() in file_path.name.lower():
                return file_path
        return None

    def _validate_npy_file(self, file_path: Path) -> bool:
        """验证numpy文件"""
        try:
            np.load(file_path)
            return True
        except:
            return False

    def _validate_tsv_file(self, file_path: Path, expected_columns: int) -> bool:
        """验证tsv文件"""
        if not file_path.exists():
            return False
        try:
            df = pd.read_csv(file_path, sep="\t", header=None)
            return len(df.columns) == expected_columns
        except:
            return False

    def _extract_tar_gz(self, file_path: Path, output_dir: Path) -> Path:
        """解压 tar.gz 文件并返回解压出的顶层路径"""
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist")

        logger.info(f"Extracting {file_path} to {output_dir}")

        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
            # 获取解压出来的顶层文件或目录列表
            members = [m.name.split("/")[0] for m in tar.getmembers() if m.name]
            # 去重，保留顶层名字
            top_level = list(set(members))

        if not top_level:
            raise ValueError(f"No files extracted from {file_path}")

        # 返回第一个顶层文件/目录路径（可根据需要改成返回所有顶层路径）
        extracted_path = output_dir / top_level[0]
        if not extracted_path.exists():
            raise ValueError(f"Expected extracted path {extracted_path} not found")

        return extracted_path

    def _convert_entity_embeddings(self, entity_emb_fpath: Path, entity_metadata_fpath: Path, 
                                 entity_id_map_fpath: Path, output_dir: Path) -> Path:
        """转换实体嵌入文件"""
        # 验证文件
        if not self._validate_npy_file(entity_emb_fpath):
            raise ValueError(f"Cannot load '{entity_emb_fpath}' or file format is incorrect")

        if not self._validate_tsv_file(entity_id_map_fpath, 2):
            raise ValueError(f"Cannot load '{entity_id_map_fpath}' or file format is incorrect")

        # 加载实体嵌入
        logger.info(f"Loading entity embeddings from '{entity_emb_fpath}'")
        entity_embeddings = np.load(entity_emb_fpath)

        # 解压并加载实体ID映射
        entity_id_map_fpath = self._extract_tar_gz(entity_id_map_fpath, output_dir)
        entity_id_map = pd.read_csv(
            entity_id_map_fpath, sep="\t", header=None, names=["idx", "entity_id"]
        )

        # 创建实体嵌入DataFrame
        entity_embeddings_df = entity_id_map.copy()
        entity_embeddings_df["embedding"] = entity_embeddings_df["idx"].apply(
            lambda x: "|".join(entity_embeddings[x].astype(str))
        )
        entity_embeddings_df.rename(columns={"entity_id": "id"}, inplace=True)

        # 添加实体类型和ID信息
        entity_embeddings_df["entity_id"] = entity_embeddings_df["id"].apply(
            lambda x: x.split("::")[1] if "::" in x else x
        )
        entity_embeddings_df["entity_type"] = entity_embeddings_df["id"].apply(
            lambda x: x.split("::")[0] if "::" in x else "unknown"
        )

        # 解压并加载实体元数据
        entity_metadata_fpath = self._extract_tar_gz(entity_metadata_fpath, output_dir)
        entity_metadata = pd.read_csv(entity_metadata_fpath, sep="\t", dtype=str)

        # 验证元数据列
        for col in ["id", "label", "name"]:
            if col not in entity_metadata.columns:
                raise ValueError(f"The entity metadata file should have the column '{col}'")

        entity_metadata["node_id"] = entity_metadata["label"] + "::" + entity_metadata["id"]

        # 合并嵌入和元数据
        logger.info("Annotating entity embeddings with metadata...")
        merged = entity_embeddings_df.merge(entity_metadata, left_on="id", right_on="node_id")

        # 处理未找到的记录
        unfound_records = entity_embeddings_df[
            ~entity_embeddings_df["id"].isin(merged["node_id"])
        ]
        if len(unfound_records) > 0:
            unfound_records.rename(columns={"id": "entity_name"}, inplace=True)
            unfound_records.to_csv(output_dir / "unfound_records.tsv", sep="\t", index=False)
            logger.info(f"Found {len(unfound_records)} unfound records")

        # 整理最终输出
        merged = merged.rename(columns={"name": "entity_name", "node_id": "embedding_id"})
        merged = merged[["embedding_id", "entity_id", "entity_type", "entity_name", "embedding"]]

        # 保存文件
        entity_embedding_fpath = output_dir / "entity_embeddings.tsv"
        merged.to_csv(entity_embedding_fpath, sep="\t", index=False)
        logger.info(f"Saved entity embeddings to '{entity_embedding_fpath}'")

        return entity_embedding_fpath

    def _convert_relation_embeddings(self, relation_emb_fpath: Path, 
                                   relation_type_id_map_fpath: Path, output_dir: Path) -> Path:
        """转换关系嵌入文件"""
        # 验证文件
        if not self._validate_npy_file(relation_emb_fpath):
            raise ValueError(f"Cannot load '{relation_emb_fpath}' or file format is incorrect")

        if not self._validate_tsv_file(relation_type_id_map_fpath, 2):
            raise ValueError(f"Cannot load '{relation_type_id_map_fpath}' or file format is incorrect")

        # 加载关系嵌入
        logger.info(f"Loading relation embeddings from '{relation_emb_fpath}'")
        relation_type_embeddings = np.load(relation_emb_fpath)

        # 解压并加载关系类型ID映射
        relation_type_id_map_fpath = self._extract_tar_gz(relation_type_id_map_fpath, output_dir)
        relation_type_id_map = pd.read_csv(
            relation_type_id_map_fpath, sep="\t", header=None, names=["idx", "relation_type_id"]
        )

        # 创建关系嵌入DataFrame
        relation_type_embeddings_df = relation_type_id_map.copy()
        relation_type_embeddings_df["embedding"] = relation_type_embeddings_df["idx"].apply(
            lambda x: "|".join(relation_type_embeddings[x].astype(str))
        )
        relation_type_embeddings_df.rename(columns={"relation_type_id": "id"}, inplace=True)

        # 保存文件
        relation_embedding_fpath = output_dir / "relation_type_embeddings.tsv"
        relation_type_embeddings_df.to_csv(relation_embedding_fpath, sep="\t", index=False)
        logger.info(f"Saved relation embeddings to '{relation_embedding_fpath}'")

        return relation_embedding_fpath
    
    def load_model_config(self, model_dir: str) -> Dict[str, Any]:
        """Load model config from model directory"""
        config_file = os.path.join(model_dir, "config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Config file {config_file} not found")

    def download_and_convert(self, run_id: str) -> Dict[str, str]:
        """
        下载并转换模型的完整流程
        
        Args:
            run_id (str): 模型运行ID
            
        Returns:
            Dict[str, str]: 转换后的文件路径字典
        """
        try:
            # 下载模型
            model_dir = self.download_model(run_id)
            # 转换模型文件
            converted_files = self.convert_model_files(model_dir)

            logger.info("✅ Successfully completed download and conversion process")
            return converted_files

        except Exception as e:
            logger.error(f"❌ Failed to download and convert model {run_id}: {e}")
            raise


# 使用示例
if __name__ == "__main__":
    # 创建Model实例
    model = Model("biomedgps-kge-v1")
    
    # 获取所有模型信息
    models = model.get_all_models()
    print(f"Found {len(models)} models")
    
    # 下载并转换特定模型
    run_id = "sdihyrpu"
    converted_files = model.download_and_convert(run_id)
    print(f"Converted files: {converted_files}")
