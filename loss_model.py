import torch.nn as nn
from typing import Dict, List
from structure import BRegGraph
from measure import OHEMLoss
from dgl import DGLGraph
from torch import Tensor


class LossModel(nn.Module):
    def __init__(self, model: Dict, criterion: Dict, device, vocab: int, class_num: int):
        super().__init__()
        self._model: BRegGraph = BRegGraph(**model, vocab=vocab, class_num=class_num)
        self._model = self._model.to(device)
        self._criterion: OHEMLoss = OHEMLoss(**criterion)
        self._criterion = self._criterion.to(device)
        self._device = device

    def forward(self,
                graphs: DGLGraph,
                labels: Tensor,
                texts: Tensor,
                lengths: Tensor,
                node_factors: Tensor,
                edge_factors: Tensor,
                node_sizes: List,
                edge_sizes: List,
                training: bool = True):
        # send data_v1 to device
        graphs = graphs.to(self._device)
        nodes = graphs.ndata['feat'].to(self._device)
        edges = graphs.edata['feat'].to(self._device)
        texts = texts.to(self._device)
        lengths = lengths.to(self._device)
        labels = labels.to(self._device)
        node_factors = node_factors.to(self._device)
        edge_factors = edge_factors.to(self._device)

        # prediction
        score: Tensor = self._model(graphs,
                                    nodes,
                                    edges,
                                    texts,
                                    lengths,
                                    node_factors,
                                    edge_factors,
                                    node_sizes,
                                    edge_sizes)
        # calc loss
        if training:
            loss: Tensor = self._criterion(score, labels)
            return score, loss
        return score

    def predict(self,
                graphs: DGLGraph,
                texts: Tensor,
                lengths: Tensor,
                node_factors: Tensor,
                edge_factors: Tensor,
                node_sizes: List,
                edge_sizes: List):
        # send data_v1 to device
        graphs = graphs.to(self._device)
        nodes = graphs.ndata['feat'].to(self._device)
        edges = graphs.edata['feat'].to(self._device)
        texts = texts.to(self._device)
        lengths = lengths.to(self._device)
        node_factors = node_factors.to(self._device)
        edge_factors = edge_factors.to(self._device)

        # prediction
        score: Tensor = self._model(graphs,
                                    nodes,
                                    edges,
                                    texts,
                                    lengths,
                                    node_factors,
                                    edge_factors,
                                    node_sizes,
                                    edge_sizes)
        return score
