import os
import os.path as osp
import random
import torch
import subprocess
import torch.distributed as dist
import numpy as np


def check_folders(work_path):
    log_path = osp.join(work_path, 'logs')
    model_path = osp.join(work_path, 'checkpoints')
    folders = [work_path, log_path, model_path]
    if is_main_process():
        for f in folders:
            if not osp.exists(f):
                os.makedirs(f)
    return log_path, model_path


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


def init_dist_slurm(backend='nccl', port=None):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def set_random_seed(seed):
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(save_path, state, save_epoch, is_best):
    checkpoint_path = osp.join(save_path, f'checkpoint{save_epoch:04}.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        best_checkpoint_path = osp.join(save_path, 'best.pth.tar')
        torch.save(state, best_checkpoint_path)