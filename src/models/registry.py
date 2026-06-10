"""
模型注册器：通过装饰器将模型类注册到全局字典，支持配置驱动的工厂构建。

用法:
    @register_model("MyModel")
    class MyModel(nn.Module): ...

    model = build_model({"model": {"name": "MyModel", "params": {...}}})
"""

from typing import Dict, Any, Type
import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """装饰器：将模型类注册到 MODEL_REGISTRY"""
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise KeyError(f"模型 '{name}' 已注册，请使用不同的名称。")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    从配置字典构建模型实例。

    期望结构:
        {"model": {"name": "ResCoordUNet", "params": {"in_channels": 3, ...}}}
    """
    model_cfg = config["model"]
    name = model_cfg["name"]
    params = model_cfg.get("params", {})

    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"未知模型 '{name}'。可用模型: {available}")

    return MODEL_REGISTRY[name](**params)


def list_models() -> list:
    """列出所有已注册的模型名称"""
    return list(MODEL_REGISTRY.keys())
