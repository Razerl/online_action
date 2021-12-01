import torch
import torch.utils.data as data
from .dataset import HaierDataset


def build_dataset(cfg, phase):
    return HaierDataset(cfg, phase)


def make_data_loader(cfg, phase='train'):
    dataset = build_dataset(cfg, phase)
    sampler = data.distributed.DistributedSampler(dataset)

    data_loaders = data.DataLoader(dataset=dataset,
                                   batch_size=cfg.solver.batch_size, num_workers=cfg.dataset.num_workers,
                                   pin_memory=True, sampler=sampler, drop_last=True)
    return data_loaders
