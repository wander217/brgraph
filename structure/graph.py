import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Tuple
from dgl import DGLGraph
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence


class GraphNorm(nn.Module):
    def __init__(self,
                 feature_num: int,
                 eps: float = 1e-5):
        super().__init__()
        self._eps: float = eps
        self._gamma: nn.Parameter = nn.Parameter(torch.ones(feature_num))
        self._beta: nn.Parameter = nn.Parameter(torch.zeros(feature_num))

    def _norm(self, x: Tensor):
        """
            :param x: a tensor with feature_num feature at last
            :return:: normed tensor of which size is the same as original tensor
        """
        # calc mean of overall graph
        mean: Tensor = x.mean(dim=0, keepdim=True)
        # calc var of overall graph
        var: Tensor = x.std(dim=0, keepdim=True)
        return (x - mean) / (var + self._eps)

    def forward(self, x: Tensor, size: List):
        """
            :param x: a graph batch
            :param size: size of each graph in batch
            :return:
        """
        # split graph sequence to a list containing graph
        graph_batch: Tensor = torch.split(x, size)
        # norm each graph inside list
        normed_graph_batch: List = [self._norm(graph)
                                    for graph in graph_batch]
        # flatten graph list
        y: Tensor = torch.cat(normed_graph_batch, dim=0)
        y = self._gamma * y + self._beta
        return y


class Dense(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self._fc: nn.Module = nn.Sequential(
            nn.LayerNorm(in_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel, out_channel, bias=True))

    def forward(self, x: Tensor) -> Tensor:
        y: Tensor = self._fc(x)
        return y


def message(edge) -> Dict:
    Bv_j = edge.src['Bv']
    # src_node_feature + connected_edge_feature + target_node_feature
    # Aggregate feature through all adjacent nodes
    e_ij = edge.src['Dv'] + edge.data['Ce'] + edge.dst['Ev']
    score = torch.sigmoid(e_ij)
    # edge.data_v1['e'] = e_ij
    return {'Bv_j': Bv_j, 'score': score}


def reduce(nodes) -> Dict:
    Av_i = nodes.data['Av']
    Bv_j = nodes.mailbox['Bv_j']
    score = nodes.mailbox['score']
    h = Av_i + torch.sum(score * Bv_j, dim=1) / (torch.sum(score, dim=1) + 1e-6)
    return {'v': h}


class GatedGCN(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 dropout: float):
        super().__init__()
        self._fc: nn.ModuleList = nn.ModuleList([
            nn.Linear(in_channel, out_channel, bias=True)
            for _ in range(5)
        ])
        # graph normalization:
        # 0: normalization for node
        # 1: normalization for edge
        self._norm: nn.ModuleList = nn.ModuleList([
            GraphNorm(out_channel) for _ in range(2)
        ])
        self._residual: bool = (in_channel == out_channel)
        self._dropout: float = dropout

    def _do_norm(self,
                 feature_i1: Tensor,
                 feature_i: Tensor,
                 num: List,
                 t: int):
        feature_i1: Tensor = F.relu(self._norm[t](feature_i1, num))
        if self._residual:
            feature_i1 = feature_i1 + feature_i
        feature_i1 = F.dropout(feature_i1,
                               self._dropout,
                               training=self.training)
        return feature_i1

    def forward(self,
                graph: DGLGraph,
                node_feature_i: Tensor,
                edge_feature_i: Tensor,
                node_factor: Tensor,
                edge_factor: Tensor,
                node_num: List,
                edge_num: List) -> Tuple:
        """
        :param graph: batch of graph at layer i
        :param node_feature_i: node feature at layer i
        :param edge_feature_i: edge feature at layer i
        :param node_factor: factor of batch node at layer i
        :param edge_factor: factor of edge node at layer i
        :param node_num: the number of node inside a batch at layer i
        :param edge_num: the number of edge inside a batch at layer i
        :return: node feature and edge feature at layer i + 1
        """

        # init edge feature
        graph.edata['e'] = edge_feature_i
        graph.edata['Ce'] = self._fc[4](edge_feature_i)
        # init node feature
        graph.ndata['v'] = node_feature_i
        for i, item in enumerate(['Av', 'Bv', 'Dv', 'Ev']):
            graph.ndata[item] = self._fc[i](node_feature_i)
        # aggregation overall graph
        graph.update_all(message, reduce)
        # update node and edge feature (layer i+1)
        node_feature_i1: Tensor = graph.ndata['v']
        edge_feature_i1: Tensor = graph.edata['e']

        # norm, residual and dropout
        node_feature_i1 = self._do_norm(node_feature_i1 * node_factor,
                                        node_feature_i,
                                        node_num,
                                        0)
        edge_feature_i1 = self._do_norm(edge_feature_i1 * edge_factor,
                                        edge_feature_i,
                                        edge_num,
                                        1)
        return node_feature_i1, edge_feature_i1


class Readout(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, layer_num: int):
        super().__init__()
        fcs: List = [nn.Linear(in_channel // 2 ** i,
                               in_channel // 2 ** (i + 1),
                               bias=True)
                     for i in range(layer_num)]
        fcs.append(nn.Linear(in_channel // 2 ** layer_num,
                             out_channel,
                             bias=True))
        self._fcs = nn.ModuleList(fcs)
        self._layer_num: int = layer_num

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = x
        for i in range(self._layer_num):
            output = F.relu(self._fcs[i](output))
        output = self._fcs[self._layer_num](output)
        return output


class BRegGraph(nn.Module):
    def __init__(self,
                 vocab: int,
                 class_num: int,
                 node_channel: int,
                 edge_channel: int,
                 hidden_channel: int,
                 dropout: float,
                 layer_num: int):
        super().__init__()
        self._text_embedding: nn.Module = nn.Embedding(vocab, hidden_channel)
        # self._node_embedding: nn.Module = nn.Linear(node_channel, hidden_channel)
        self._edge_embedding: nn.Module = nn.Linear(edge_channel, hidden_channel)
        self._layers: nn.ModuleList = nn.ModuleList([
            GatedGCN(hidden_channel,
                     hidden_channel,
                     dropout)
            for _ in range(layer_num)
        ])
        self._dense: nn.ModuleList = nn.ModuleList([
            Dense(hidden_channel + i * hidden_channel,
                  hidden_channel)
            for i in range(1, layer_num + 1)
        ])
        self._lstm: nn.Module = nn.LSTM(input_size=hidden_channel,
                                        hidden_size=hidden_channel,
                                        num_layers=2,
                                        batch_first=True,
                                        bidirectional=True)
        self._mlp: nn.Module = Readout(hidden_channel, class_num, 2)

    def _lstm_text_embedding(self,
                             texts: Tensor,
                             lengths: Tensor):
        text_embedding: Tensor = self._text_embedding(texts)  # (B, max_length, hidden_channel)
        packed_sequence: PackedSequence = pack_padded_sequence(text_embedding,
                                                               lengths.cpu(),
                                                               True,
                                                               False)
        output, (h_last, c_last) = self._lstm(packed_sequence)
        return F.normalize(h_last.mean(0))

    def _concat(self, nodes: List, i: int) -> Tensor:
        concat_node: Tensor = torch.cat(nodes, dim=1)
        output: Tensor = self._dense[i](concat_node)
        return output

    def forward(self,
                graphs: DGLGraph,
                nodes: Tensor,
                edges: Tensor,
                texts: Tensor,
                lengths: Tensor,
                node_factors: Tensor,
                edge_factors: Tensor,
                node_sizes: List,
                edge_sizes: List) -> Tensor:
        # node_embedding: Tensor = self._node_embedding(nodes)
        edge_embedding: Tensor = self._edge_embedding(edges)

        text_embedding = self._lstm_text_embedding(texts, lengths)
        nodes: Tensor = text_embedding
        edges: Tensor = edge_embedding
        all_node: List = [nodes]
        for i, conv in enumerate(self._layers):
            new_nodes, edges = conv(graphs,
                                    nodes,
                                    edges,
                                    node_factors,
                                    edge_factors,
                                    node_sizes,
                                    edge_sizes)
            all_node.append(new_nodes)
            nodes = self._concat(all_node, i)
        output: Tensor = self._mlp(nodes)
        return output
