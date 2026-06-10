"""
数据集注册器：通过装饰器将 Dataset 类注册，支持配置驱动的工厂构建。
"""

from typing import Dict, Any, Type
from torch.utils.data import Dataset

DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str):
    """装饰器：将数据集类注册到 DATASET_REGISTRY"""
    def decorator(cls):
        if name in DATASET_REGISTRY:
            raise KeyError(f"数据集 '{name}' 已注册。")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def build_dataset(config: Dict[str, Any], split: str = "train"):
    """
    从配置字典构建数据集。

    split: "train" | "val" | "test"

    期望配置结构:
        dataset:
          name: "ISIC2018"
          params: {img_size: [224, 224], val_split: 0.2}
          paths: {train_images: "./dataset/train/images", ...}
          augmentation: {enabled: true, flip_prob: 0.5, ...}
    """
    ds_cfg = config["dataset"]
    name = ds_cfg["name"]
    paths = ds_cfg.get("paths", {})
    params = ds_cfg.get("params", {})

    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise KeyError(f"未知数据集 '{name}'。可用数据集: {available}")

    ds_cls = DATASET_REGISTRY[name]
    return ds_cls.from_config(config, split)


def list_datasets() -> list:
    """列出所有已注册的数据集名称"""
    return list(DATASET_REGISTRY.keys())
