import torch
from torch import Tensor
import numpy as np
from sklearn.metrics import confusion_matrix


class Accurate:
    def __init__(self):
        pass

    def __call__(self, pred: Tensor, target: Tensor, label_num: int):
        target_matrix: np.ndarray = target.cpu().detach().numpy()
        pred = torch.log_softmax(pred, dim=1)
        pred_matrix: np.ndarray = np.argmax(pred.cpu().detach().numpy(), axis=1)
        cm: np.ndarray = confusion_matrix(target_matrix, pred_matrix).astype(np.float32)
        recall: np.ndarray = np.zeros(label_num, dtype=np.float32)
        precision: np.ndarray = np.zeros(label_num, dtype=np.float32)
        h_mean: np.ndarray = np.zeros(label_num, dtype=np.float32)

        for r in range(cm.shape[0]):
            cluster: np.ndarray = np.where(target_matrix == r)[0]
            if cluster.shape[0] != 0:
                recall[r] = cm[r, r] / float(cluster.shape[0])
                lower = np.sum(cm[:, r])
                precision[r] = cm[r, r] / lower if lower > 0 else 0.
                lower = recall[r] + precision[r]
                h_mean[r] = 2 * recall[r] * precision[r] / lower if lower > 0 else 0.
        return recall, precision, h_mean
