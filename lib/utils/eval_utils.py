import torch
from torch import nn


class SetEvaluation(nn.Module):
    def __init__(self, cfg, metrics=None):
        super(SetEvaluation, self).__init__()
        if metrics is None:
            metrics = ['acc']
        self.model = cfg.model.architecture
        self.metrics = metrics

    def get_metrics(self, metric, outputs, targets):
        metric_map = {
            'acc': self.accuracy,
        }
        return metric_map[metric](outputs, targets, name=metric)

    def accuracy(self, output, target, name, topk=(1,5)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return {name: res}

    def forward(self, outputs, targets):
        res = None
        metrics = {}
        if self.model == 'OadTR':
            enc_score_p0, dec_scores = outputs
            class_h_target, dec_target = targets
            outputs = {
                'acc': enc_score_p0.data,
            }
            targets = {
                'labels_encoder': torch.argmax(class_h_target, dim=1),
            }
            for metric in self.metrics:
                metrics.update(self.get_metrics(metric, outputs[metric], targets[metric]))
        return res
