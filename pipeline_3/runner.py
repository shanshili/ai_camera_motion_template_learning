"""管线编排器：按 config.pipeline.stages 顺序运行各阶段（带横幅与计时）。"""

import time

from .context import PipelineContext
from .stage import get_stage
from .log import log, stage_banner, stage_done
from . import stages  # noqa: F401 触发阶段注册


def run_pipeline(config: dict) -> PipelineContext:
    ctx = PipelineContext(
        config=config,
        input_path=config["io"]["input"],
        output_path=config["io"]["output"],
    )
    stage_names = config.get("pipeline", {}).get(
        "stages", ["probe", "analysis", "shotplan", "camera", "render"])
    log(f"=== 管线启动：{' -> '.join(stage_names)} ===")
    t_all = time.time()
    for i, name in enumerate(stage_names, 1):
        stage_banner(name, i, len(stage_names))
        t0 = time.time()
        ctx = get_stage(name)().run(ctx)
        stage_done(name, t0)
    log(f"=== 管线完成，总用时 {time.time() - t_all:.1f}s ===")
    return ctx
