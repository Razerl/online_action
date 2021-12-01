import torch
from .transformer_models.ViT import VisionTransformer_v3

_ARCHITECTURES = {"OadTR": VisionTransformer_v3}


def build_model(cfg):
    model_arch = _ARCHITECTURES[cfg.model.architecture]
    model = model_arch(cfg)

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=True
    )

    return model
