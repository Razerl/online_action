import argparse
import time
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from pathlib import Path
from tensorboardX import SummaryWriter
from lib.configs import cfg
from lib.utils.common import *
from lib.utils.logger import setup_logger
from lib.models.build import build_model
from lib.solver.build import make_optimizer, make_lr_scheduler
from lib.data.build import make_data_loader


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


def train(cfg, logger, tf_writer):
    model = build_model(cfg.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))

    data_loader = make_data_loader(cfg, phase='train')

    optimizer = make_optimizer(cfg.solver, model)
    scheduler = make_lr_scheduler(cfg.solver, len(data_loader), optimizer)

    if cfg.model.frozen_weights is not None:
        checkpoint = torch.load(cfg.model.frozen_weights, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    if cfg.model.resume:
        if os.path.isfile(cfg.model.resume):
            logger.info(("=> loading checkpoint '{}'".format(cfg.model.resume)))
            checkpoint = torch.load(cfg.model.resume, map_location='cpu')
            cfg.training.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.training.evaluate, checkpoint['epoch'])))
        else:
            logger.info(("=> no checkpoint found at '{}'".format(cfg.model.resume)))

    if cfg.model.tune_from:
        logger.info(("=> fine-tuning from '{}'".format(cfg.model.tune_from)))
        sd = torch.load(cfg.model.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        keys1, keys2 = set(list(sd.keys())), set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        logger.info('#### Notice: keys that failed to load: {}'.format(set_diff))
        model_dict.update(sd)
        model.load_state_dict(model_dict, strict=True)

    if cfg.training.evaluate:
        print('start testing for one epoch !!!')
        with torch.no_grad():
            test_stats = evaluate(model, criterion, data_loader_val, device, logger, args)
        return test_stats


def evaluate():
    return


def main():
    args = parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    # set cudnn_benchmark
    cudnn.benchmark = True

    if cfg.get('output_dir', None) is None:
        # use config filename as default output_dir if cfg.output_dir is None
        cfg.output_dir = osp.join('./outputs', osp.splitext(osp.basename(args.config))[0])
    # create output_dir
    output_dir = Path(osp.abspath(cfg.output_dir))
    if is_main_process() and not osp.exists(output_dir):
        os.makedirs(output_dir)

    if args.resume_from is not None:
        cfg.model.resume_from = args.resume_from

    # init distributed env first, since logger depends on the dist info.
    init_dist(cfg.common.dist)

    # dump config
    cfg.dump(osp.join(cfg.output_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.output_dir, 'log', f'{timestamp}.log')
    logger = setup_logger(output=log_file, distributed_rank=dist.get_rank(), name=f'OAD')

    # create Tensorboard
    tf_writer = SummaryWriter(log_dir=osp.join(cfg.output_dir, 'log'))

    # log some basic info
    logger.info(f'Distributed training:')
    logger.info(f'Config: {cfg.pretty_text}')

    # fix the seed
    set_random_seed(args.seed)

    train(cfg, logger, tf_writer)


if __name__ == '__main__':
    main()
