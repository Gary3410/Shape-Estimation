import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction=reduction,
        ignore_index=ignore_index)

    return loss

def _expand_onehot_labels(labels, label_weights=None, label_channels=None):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

def binary_cross_entropy(pred,
                         label,
                         use_sigmoid=False,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    if use_sigmoid:
        loss = F.binary_cross_entropy_with_logits(
            pred, label.float(), weight=class_weight, reduction=reduction)
    else:
        loss = F.binary_cross_entropy(
            pred, label.float(), weight=class_weight, reduction=reduction)
    # do the reduction for the weighted loss
    # print(loss)
    return loss


class AffinityLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cls_criterion = binary_cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        unary_term = self.cls_criterion(
            cls_score,
            label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        diagonal_matrix = (1 - torch.eye(label.size(1))).to(label.get_device())
        vtarget = diagonal_matrix * label

        recall_part = torch.sum(cls_score * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = self.cls_criterion(
            recall_part,
            recall_label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        spec_part = torch.sum((1 - cls_score) * (1 - label), dim=2)
        denominator = torch.sum(1 - label, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = self.cls_criterion(
            spec_part,
            spec_label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        precision_part = torch.sum(cls_score * vtarget, dim=2)
        denominator = torch.sum(cls_score, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = self.cls_criterion(
            precision_part,
            precision_label,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        global_term = recall_loss + spec_loss + precision_loss
        #unary_term = torch.sum(unary_term, dim=2)
        loss_cls = self.loss_weight * (unary_term + global_term)
        return loss_cls