from ...schedulers import RASFlowMatchEulerDiscreteScheduler
from ...schedulers import ChetanRASFlowMatchEulerDiscreteScheduler
from ...schedulers import VayunRASFlowMatchEulerDiscreteScheduler
from ...modules.attention_processor import RASJointAttnProcessor2_0
from ...modules.stable_diffusion_3.transformer_forward import ras_forward
from ...utils import ras_manager

def update_sd3_pipeline(pipeline):
    if ras_manager.MANAGER.method == "chetan":
        scheduler = ChetanRASFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif ras_manager.MANAGER.method == "viyom":
        scheduler = VayunRASFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif ras_manager.MANAGER.method == "default":
        scheduler = RASFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        return Error("Unknown RAS method")
    pipeline.scheduler = scheduler
    pipeline.transformer.forward = ras_forward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.transformer_blocks:
        block.attn.set_processor(RASJointAttnProcessor2_0())
    return pipeline
