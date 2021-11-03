import argparse
import random
import time
import torch
from lib.configs import cfg
from lib.utils.common import *
from lib.utils.logger import setup_logger


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

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    model = transformer_models.VisionTransformer_v3(args=args, img_dim=args.enc_layers,  # VisionTransformer_v3
                                                    patch_dim=args.patch_dim,
                                                    out_dim=args.numclass,
                                                    embedding_dim=args.embedding_dim,
                                                    num_heads=args.num_heads,
                                                    num_layers=args.num_layers,
                                                    hidden_dim=args.hidden_dim,
                                                    dropout_rate=args.dropout_rate,
                                                    attn_dropout_rate=args.attn_dropout_rate,
                                                    num_channels=args.dim_feature,
                                                    positional_encoding_type=args.positional_encoding_type
                                                    )
    model.to(device)

    loss_need = [
        'labels_encoder',
        'labels_decoder',
    ]
    criterion = utl.SetCriterion(num_classes=args.numclass, losses=loss_need, args=args).to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=True,
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
    elif args.dataparallel:
        args.gpu = '0,1,2,3'
        model = nn.DataParallel(model, device_ids=[int(iii) for iii in args.gpu.split(',')])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params: {}'.format(n_parameters))
    # logger.output_print(args)

    dataset_train = DataLayer(phase='train', args=args)
    dataset_val = DataLayer(phase='test', args=args)

    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(dataset_train,
                                   batch_size=args.batch_size, sampler=sampler_train,
                                   pin_memory=True, num_workers=args.num_workers, drop_last=True)
    data_loader_val = DataLoader(dataset_val,
                                 batch_size=args.batch_size, sampler=sampler_val,
                                 pin_memory=True, num_workers=args.num_workers, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 )
    lr_scheduler = get_scheduler(optimizer, len(data_loader_train), args)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(os.path.join(args.output_dir, args.store_name))
    if utils.is_main_process() and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print('checkpoint: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    if args.eval:
        print('start testing for one epoch !!!')
        with torch.no_grad():
            test_stats = test_one_epoch(model, criterion, data_loader_val, device, logger, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        train_stats = train(
            model, criterion, data_loader_train, optimizer, device, epoch,
            logger, lr_scheduler, args.clip_max_norm)

        if dist.get_rank() == 0:
            tf_writer.add_scalar('loss/train', train_stats['loss'], epoch)
            tf_writer.add_scalar('loss_encoder/train', train_stats['label_encoder'], epoch)
            tf_writer.add_scalar('loss_decoder/train', train_stats['label_decoder'], epoch)
            tf_writer.add_scalar('acc/train_top1', train_stats['top1'], epoch)
            tf_writer.add_scalar('acc/train_top5', train_stats['top5'], epoch)
            tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

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
