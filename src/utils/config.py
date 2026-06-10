"""配置加载器：从 YAML 文件加载并校验配置"""

import os
import yaml
from typing import Any, Dict


class ConfigLoader:
    """YAML 配置加载器，支持点号访问嵌套字段"""

    def __init__(self, yaml_path: str):
        self._path = yaml_path
        self.cfg = self._load(yaml_path)
        self._validate()

    def _load(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate(self):
        """校验必填字段"""
        required_top = ["experiment", "model", "dataset", "training"]
        for key in required_top:
            if key not in self.cfg:
                raise KeyError(f"配置文件缺少必填字段: '{key}'")

        # 校验模型
        model = self.cfg["model"]
        if "name" not in model:
            raise KeyError("model 段缺少 'name'")

        # 校验数据集
        ds = self.cfg["dataset"]
        if "name" not in ds:
            raise KeyError("dataset 段缺少 'name'")
        if "paths" not in ds:
            raise KeyError("dataset 段缺少 'paths'")

        # 校验训练
        train = self.cfg["training"]
        for required in ["loss", "optimizer"]:
            if required not in train:
                raise KeyError(f"training 段缺少 '{required}'")
            if "name" not in train[required]:
                raise KeyError(f"training.{required} 段缺少 'name'")

    def get(self, key: str, default=None):
        """支持点号分隔的嵌套访问，如 get('training.loss.params.dice_weight')"""
        keys = key.split(".")
        val = self.cfg
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def __getitem__(self, key: str):
        return self.cfg[key]

    def __contains__(self, key: str):
        return key in self.cfg

    def __repr__(self):
        return f"ConfigLoader({self._path})"
