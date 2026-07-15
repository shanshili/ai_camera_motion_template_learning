"""
处理阶段基类与注册表
====================
每个阶段是一个 Stage 子类，实现 run(ctx)。
config.pipeline.stages 里的名字通过注册表映射到具体类。
"""

from abc import ABC, abstractmethod

from .context import PipelineContext

_REGISTRY = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_stage(name):
    if name not in _REGISTRY:
        raise KeyError(f"未注册的阶段: {name}（已注册: {list(_REGISTRY)}）")
    return _REGISTRY[name]


class Stage(ABC):
    name = "stage"

    @abstractmethod
    def run(self, ctx: PipelineContext) -> PipelineContext:
        ...
