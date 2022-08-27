import os
import logging
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List
from collections import OrderedDict


def remove_space(txt: str):
    return txt.strip().strip("\r\t").strip("\n")


class Averager:
    def __init__(self):
        self.value: float = 0
        self.num: int = 0

    def update(self, value: float, num: int):
        self.value += value
        self.num += num

    def calc(self):
        if self.num == 0:
            return 0
        return self.value / self.num

    def clear(self):
        self.value = 0
        self.num = 0


class Checkpoint:
    def __init__(self, workspace: str, resume: str):
        self._workspace: str = workspace
        if not os.path.isdir(workspace):
            os.mkdir(workspace)
        self._resume: str = resume.strip()

    def save_last(self,
                  epoch: int,
                  model: nn.Module,
                  optimizer: optim.Optimizer):
        last_path: str = os.path.join(self._workspace, "last.pth")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, last_path)

    def save_model(self, model: nn.Module, epoch: int) -> Any:
        path: str = os.path.join(self._workspace, "checkpoint_{}.pth".format(epoch))
        torch.save({"model": model.state_dict()}, path)

    def load(self, device=torch.device('cpu')):
        if isinstance(self._resume, str) and bool(self._resume):
            data: OrderedDict = torch.load(self._resume, map_location=device)
            model: OrderedDict = data.get('model')
            optimizer: OrderedDict = data.get('optimizer')
            epoch: int = data.get('epoch')
            return model, optimizer, epoch

    def load_path(self, path: str, device=torch.device('cpu')) -> OrderedDict:
        data: OrderedDict = torch.load(path, map_location=device)
        assert 'model' in data
        model: OrderedDict = data.get('model')
        return model


class Logger:
    def __init__(self, workspace: str, level: str):
        if not os.path.isdir(workspace):
            os.mkdir(workspace)
        self._workspace: str = workspace

        self._level: int = logging.INFO if level == "INFO" else logging.DEBUG
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        self._logger = logging.getLogger("message")
        self._logger.setLevel(self._level)

        file_handler = logging.FileHandler(os.path.join(self._workspace, "ouput.log"))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self._level)
        self._logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(self._level)
        self._logger.addHandler(stream_handler)
        self._time: float = time.time()
        self._save_path: str = os.path.join(self._workspace, "metric.txt")

    def report_time(self, name: str):
        current: float = time.time()
        self._write(name + " - time: {}".format(current - self._time))

    def report_metric(self, name: str, metric: Dict):
        self.report_delimiter()
        self.report_time(name)
        keys: List = list(metric.keys())
        for key in keys:
            self._write("\t- {}: {}".format(key, metric[key]))
        self.report_delimiter()
        self.report_newline()

    def write(self, metric: Dict):
        with open(self._save_path, 'a', encoding='utf=8') as f:
            f.write(json.dumps(metric))
            f.write("\n")

    def report_delimiter(self):
        self._write("-" * 33)

    def report_newline(self):
        self._write("")

    def _write(self, message: str):
        if self._level == logging.INFO:
            self._logger.info(message)
            return
        self._logger.debug(message)
