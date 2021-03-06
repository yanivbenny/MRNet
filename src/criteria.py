import torch
import torch.nn.functional as F


def calculate_acc(output, target):
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct * 100.0 / target.size()[0]


def calculate_correct(output, target):
    pred = output.data.max(1)[1]
    correct = pred.eq(target.data).cpu().sum().numpy()
    return correct


def contrast_loss(output, target, reduction='mean', weighted_loss=False):
    labels = torch.zeros_like(output)
    labels.scatter_(1, target.view(-1, 1), 1.0)
    weights = (1 + 6 * labels) / 7 if weighted_loss else None

    return F.binary_cross_entropy_with_logits(output, labels, weight=weights, reduction=reduction)


def type_loss(meta_pred, meta_target):
    meta_target_loss = None
    if meta_target is not None:
        meta_target_loss = F.binary_cross_entropy_with_logits(meta_pred, meta_target)
    return meta_target_loss