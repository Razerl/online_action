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
from lib.engine.inference import evaluate
from lib.engine.trainer import train
from lib.utils.loss import SetCriterion


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

    # ----------------------------------prepare---------------------------------------------
    model = build_model(cfg.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))

    data_loader_train = make_data_loader(cfg, phase='train')
    data_loader_val = make_data_loader(cfg, phase='test')

    optimizer = make_optimizer(cfg.solver, model)
    scheduler = make_lr_scheduler(cfg.solver, len(data_loader_train), optimizer)

    loss_need = [
        'labels_encoder',
        'labels_decoder',
    ]
    criterion = SetCriterion(cfg, losses=loss_need)

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
        logger.info('start testing for one epoch !!!')
        with torch.no_grad():
            test_stats = inference(model)
        return test_stats

    # --------------------------------------training-----------------------------------------

    for epoch in range(cfg.training.start_epoch, cfg.training.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        train_meters = train(cfg, model, criterion, data_loader_train, optimizer, epoch,
                             logger, scheduler)

        if dist.get_rank() == 0:
            train_meters.tf_write(tf_writer, epoch, phase='train')

        if (epoch + 1) % args.eval_freq == 0:
            data_loader_val.sampler.set_epoch(epoch)
            test_stats = test(model, criterion, data_loader_val, device, logger, args)

            if utils.is_main_process():
                tf_writer.add_scalar('loss/test', test_stats['loss'], epoch)
                tf_writer.add_scalar('loss_encoder/test', test_stats['label_encoder'], epoch)
                tf_writer.add_scalar('loss_decoder/test', test_stats['label_decoder'], epoch)
                tf_writer.add_scalar('acc/test_top1', test_stats['top1'], epoch)
                tf_writer.add_scalar('acc/test_top5', test_stats['top5'], epoch)

                prec1 = test_stats['top1']
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info(("Best Prec@1: '{}'".format(best_prec1)))

                if args.root_model:
                    checkpoint_paths = [output_dir / f'checkpoint{epoch:04}.pth']
                    if is_best:
                        checkpoint_paths.append(output_dir / 'checkpoint.pth')
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)


if __name__ == '__main__':
    main()
