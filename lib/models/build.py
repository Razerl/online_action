from .transformer_models.ViT import VisionTransformer_v3

_ARCHITECTURES = {"OadTR": VisionTransformer_v3}


def build_model(cfg):
    model_arch = _ARCHITECTURES[cfg.model.architecture]
    return model_arch(cfg)