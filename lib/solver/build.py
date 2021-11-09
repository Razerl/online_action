import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from .lr_scheduler import GradualWarmupScheduler


def make_optimizer(cfg, model):
    if cfg.optimizer == 'sgd':
        policies = model.get_optim_policies
        optimizer = torch.optim.SGD(policies, cfg.lr,
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
        return optimizer

    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay,
                                     )
        return optimizer
    else:
        ValueError('Unknown optimizer type')


def make_lr_scheduler(cfg, n_iter_per_epoch, optimizer):
    if "cosine" in cfg.lr_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.00001,
            T_max=(cfg.epochs - cfg.warmup_epoch) * n_iter_per_epoch)
    elif "muti_step" in cfg.lr_scheduler:
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=cfg.lr_decay_rate,
            milestones=[(m - cfg.warmup_epoch) * n_iter_per_epoch for m in cfg.lr_steps])
    elif "step" in cfg.lr_scheduler:
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=(cfg.lr_drop - cfg.warmup_epoch) * n_iter_per_epoch,
            gamma=cfg.lr_decay_rate)
    else:
        raise NotImplementedError(f"scheduler {cfg.lr_scheduler} not supported")

    if cfg.warmup_epoch != 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=cfg.warmup_multiplier,
            after_scheduler=scheduler,
            warmup_epoch=cfg.warmup_epoch * n_iter_per_epoch)

    return scheduler
