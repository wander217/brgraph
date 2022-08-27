import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def ohem(pred: Tensor, label: Tensor, factor: int) -> Tensor:
    pred_matrix: np.ndarray = pred.cpu().detach().numpy()
    label_matrix: np.ndarray = label.cpu().detach().numpy()
    # be sure pos / neg = 1/3
    pos_num: int = sum(label_matrix != 0)
    neg_sum: int = pos_num * factor
    # find label being not other
    pred_value: np.ndarray = pred_matrix[:, 1:].max(1)
    # sort decent by pred_value predicted other
    sorted_neg_score: np.ndarray = np.sort(-pred_value[label_matrix == 0])
    # if other > limit, set threshold to remove
    if sorted_neg_score.shape[0] > neg_sum:
        threshold = -sorted_neg_score[neg_sum - 1]
        # if label is not other , put true
        # if label is other and pred_value > thresh, put true
        mask = ((pred_value >= threshold) | (label_matrix != 0))
    else:
        # put all true
        mask = (label_matrix != -1)
    return torch.from_numpy(mask)


class OHEMLoss(nn.Module):
    def __init__(self, factor: int, ignore_index: int = -100):
        super().__init__()
        self._factor: int = factor
        self._ignore_index: int = ignore_index
        self._loss: nn.Module = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        masked_label: Tensor = label.clone()
        # if label is imbalance , reduce other label
        mask: Tensor = ohem(pred, label, self._factor).to(pred.device)
        masked_label[mask == False] = self._ignore_index
        # calc loss
        loss: Tensor = self._loss(pred, masked_label.long())
        return loss
