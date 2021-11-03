
_ARCHITECTURES = {"OadTR": }


def build_model(cfg):
    model_arch = _ARCHITECTURES[cfg.model.architecture]
    return model_arch(cfg)