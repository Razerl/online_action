import argparse
import time
import torch
from lib.configs import cfg
from lib.utils.common import *
from lib.utils.logger import setup_logger
from lib.models.build import build_model
from lib.solver.build import make_optimizer, make_lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train a online action recognizer')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help=('whether to test the best checkpoint (if applicable) after '
              'training'))
    parser.add_argument('--seed', type=int, default=20, help='random seed')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def train(cfg, logger):
    model = build_model(cfg.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    optimizer = make_optimizer(cfg.solver, model)
    scheduler = make_lr_scheduler(cfg.solver, len(data_loader), optimizer)


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = True

    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    # init distributed env first, since logger depends on the dist info.
    init_dist(cfg.common.dist)

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = setup_logger(output=log_file, distributed_rank=torch.distributed.get_rank(), name=f'OAD')

    # log some basic info
    logger.info(f'Distributed training:')
    logger.info(f'Config: {cfg.pretty_text}')

    # fix the seed
    set_random_seed(args.seed)

    train(cfg, logger)


if __name__ == '__main__':
    main()
