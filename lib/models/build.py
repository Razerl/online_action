import torch
from .transformer_models.ViT import VisionTransformer_v3

_ARCHITECTURES = {"OadTR": VisionTransformer_v3}


def build_model(cfg):
    model_arch = _ARCHITECTURES[cfg.model.architecture]
    model = model_arch(cfg)

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[cfg.local_rank],
        broadcast_buffers=True,
        find_unused_parameters=True
    )

    return model
