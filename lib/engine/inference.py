import time
import datetime
import torch
from lib.utils.metric_logger import MetricLogger


def validate(cfg, model, criterion, evaluation, data_loader, logger):
    model.eval()
    criterion.eval()

    logger.info("Start validating ... ")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    print_freq = cfg.trainig.training_print_freq
    start_val_time = time.time()
    end = time.time()
    with torch.no_grad():
        for iteration, (inputs_val, targets) in enumerate(data_loader):
            data_time = time.time() - end
            inputs_val = inputs_val.cuda()
            if isinstance(targets, (list, tuple)):
                targets = [t.cuda() for t in targets]
            else:
                targets = targets.cuda()
            outputs = model(inputs_val)

            evaluation_dict = evaluation(outputs, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            evaluation_dict = {
                'top1': evaluation_dict['acc'][0],
                'top5': evaluation_dict['acc'][1],
            }
            evaluation_dict.pop('acc')
            meters.update(losses=losses, **loss_dict)
            meters.update(**evaluation_dict)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iteration % print_freq == 0 or iteration == max_iter - 1:
                logger.info(
                    meters.delimiter.join(
                        [
                            "iter: {iter}/{total}",
                            "eta: {eta}}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        iter=iteration,
                        total=max_iter,
                        meters=str(meters),
                    )
                )

    total_val_time = time.time() - start_val_time
    total_time_str = str(datetime.timedelta(seconds=total_val_time))
    logger.info(
        "Total validation time: {} ({:.4f} s / it)".format(
            total_time_str, total_val_time / max_iter
        )
    )
    return meters
