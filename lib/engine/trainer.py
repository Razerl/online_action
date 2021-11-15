import torch
import time
from lib.utils.metric_logger import MetricLogger


def train(cfg, model, criterion, data_loader, optimizer, epoch,
          logger, scheduler=None, max_norm=0.):
    logger.info("Start training ... ")
    meters = MetricLogger(delimiter="  ")
    training_print_freq = cfg.trainig.training_print_freq
    num_class = cfg.model.num_class

    start_training_time = time.time()
    end = time.time()
    for i, (camera_inputs, enc_target, distance_target, class_h_target, dec_target) in enumerate(data_loader):
        data_time.update(time.time() - end)
        camera_inputs = camera_inputs.to(device)
        class_h_target = class_h_target.to(device)
        dec_target = dec_target.to(device)

        enc_score_p0, dec_scores = model(camera_inputs)

        outputs = {
            'labels_encoder': enc_score_p0,  # [128, 22]
            'labels_decoder': dec_scores.view(-1, num_class),  # [128, 8, 22]
        }
        targets = {
            'labels_encoder': class_h_target.view(-1, num_class),
            'labels_decoder': dec_target.view(-1, num_class),
        }
        prec1, prec5 = utils.accuracy(enc_score_p0.data, torch.argmax(class_h_target, dim=1), topk=(1, 5))
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        loss.update(loss_value, camera_inputs.size(0))
        loss_encoder.update(loss_dict['labels_encoder'], camera_inputs.size(0))
        loss_decoder.update(loss_dict['labels_decoder'], camera_inputs.size(0))

        top1.update(prec1.item(), camera_inputs.size(0))
        top5.update(prec5.item(), camera_inputs.size(0))

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == len(data_loader) - 1:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Label encoder {encoder.val:.4f} ({encoder.avg:.4f})\t'
                         'Label decoder {decoder.val:.4f} ({decoder.avg:.4f})'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                epoch, i, len(data_loader), batch_time=batch_time, data_time=data_time, loss=loss, top1=top1,
                top5=top5, encoder=loss_encoder, decoder=loss_decoder, lr=optimizer.param_groups[-1]['lr']))
    stats = {
        'loss': loss.avg,
        'label_encoder': loss_encoder.avg,
        'label_decoder': loss_decoder.avg,
        'top1': top1.avg,
        'top5': top5.avg
    }
    return stats
