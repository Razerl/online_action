import torch
import torch.nn.functional as F
from torch import nn
from ipdb import set_trace


class SetCriterion(nn.Module):

    def __init__(self, cfg, losses):
        """ Create the criterion.
        Parameters:
            cfg: args
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = cfg.model.num_class
        self.weight_dict = {
            'labels_encoder': cfg.loss.enc_loss_coef,
            'labels_decoder': cfg.loss.dec_loss_coef,
            'distance': cfg.loss.similar_loss_coef,
        }
        self.losses = losses
        self.ignore_index = cfg.loss.sample_cls_index
        self.weight = cfg.loss.sample_weight
        self.margin = cfg.loss.contrastive_loss_margin
        self.model = cfg.model.architecture
        self.num_class = cfg.model.num_class
        self.size_average = True
        self.log_softmax = nn.LogSoftmax(dim=1)

    def loss_ce(self, _input, targets, name):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        target = targets.float()
        if self.ignore_index >= 0:
            output = torch.sum(-target * self.log_softmax(_input), 1)
            loss_ = torch.sum(self.weight * output[target[:, self.ignore_index] != 1]) + \
                    torch.sum(output[target[:, self.ignore_index] == 1])
            loss_ce = loss_ / targets.size(0)
        else:
            output = torch.sum(-target * self.log_softmax(_input), 1)
            if self.size_average:
                loss_ce = torch.mean(output)
            else:
                loss_ce = torch.sum(output)
        if torch.isnan(loss_ce).sum() > 0:
            set_trace()
        losses = {name: loss_ce}
        return losses

    def contrastive_loss(self, output, label, name):
        """
        Contrastive loss function.
        Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """
        output1, output2 = output
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1. - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        if torch.isnan(loss_contrastive).sum() > 0:
            set_trace()
        losses = {name: loss_contrastive.double()}
        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'labels_encoder': self.ce_loss,
            'labels_decoder': self.ce_loss,
            'distance': self.contrastive_loss,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, name=loss)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        if self.model == 'OadTR':
            enc_score_p0, dec_scores = outputs
            class_h_target, dec_target = targets
            outputs = {
                'labels_encoder': enc_score_p0,
                'labels_decoder': dec_scores.view(-1, self.num_class),
            }
            targets = {
                'labels_encoder': class_h_target.view(-1, self.num_class),
                'labels_decoder': dec_target.view(-1, self.num_class),
            }
        else:
            raise NotImplementedError(f"model {self.model} not supported")
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs[loss], targets[loss]))
        return losses
