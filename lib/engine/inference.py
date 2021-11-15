


def evaluate(model, criterion, data_loader, device, logger, args):
    model.eval()
    criterion.eval()

    batch_time = utils.AverageMeter()
    loss = utils.AverageMeter()
    loss_encoder = utils.AverageMeter()
    loss_decoder = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    num_class = args.numclass
    print_freq = 40
    end = time.time()
    with torch.no_grad():
        for i, (camera_inputs_val, enc_target_val, distance_target_val, class_h_target_val, dec_target) in enumerate(data_loader):
            camera_inputs = camera_inputs_val.to(device)
            class_h_target = class_h_target_val.to(device)
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

            loss.update(loss_value, camera_inputs.size(0))
            loss_encoder.update(loss_dict['labels_encoder'], camera_inputs.size(0))
            loss_decoder.update(loss_dict['labels_decoder'], camera_inputs.size(0))

            top1.update(prec1.item(), camera_inputs.size(0))
            top5.update(prec5.item(), camera_inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 or i == len(data_loader) - 1:
                logger.info(('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Label encoder {encoder.val:.4f} ({encoder.avg:.4f})\t'
                             'Label decoder {decoder.val:.4f} ({decoder.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})').format(
                    i, len(data_loader), batch_time=batch_time, loss=loss,
                    encoder=loss_encoder, decoder=loss_decoder, top1=top1, top5=top5))

    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss '
                 '{loss.avg:.5f}, '.format(top1=top1, top5=top5, loss=loss)))
    stats = {
        'loss': loss.avg,
        'label_encoder': loss_encoder.avg,
        'label_decoder': loss_decoder.avg,
        'top1': top1.avg,
        'top5': top5.avg
    }
    return stats