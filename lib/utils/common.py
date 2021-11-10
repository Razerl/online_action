import os
import re
import random
import torch
import os.path as osp
import torch.distributed as dist
import numpy as np


def is_main_process():
    return get_rank() == 0


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_ip(ip):
    ip = re.sub('SH-IDC1-', '', ip)
    ip = ip.strip().split('-')
    ip = ip[0] + '.' + ip[1] + '.' + ip[2] + '.' + ip[3]
    return ip


def init_dist(dist_params):
    local_rank = int(os.environ['SLURM_LOCALID'])
    dist_params.local_rank = local_rank
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    ip = get_ip(os.environ['SLURM_STEP_NODELIST'])

    host_addr_full = 'tcp://' + ip + ':' + dist_params.port
    torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                         rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()


def set_random_seed(seed):
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
