import os.path
import yaml
import argparse
import torch
from measure import Accurate
import torch.optim as optim
from loss_model import LossModel
from dataset import GraphLoader, GraphDataset, GraphLabel, GraphAlphabet
from utils import Checkpoint, Averager, Logger
from typing import Dict, List, Tuple
import warnings


class Trainer:
    def __init__(self,
                 total_epoch: int,
                 start_epoch: int,
                 alphabet: Dict,
                 label: Dict,
                 loss_model: Dict,
                 optimizer: Dict,
                 train: Dict,
                 valid: Dict,
                 test: Dict,
                 checkpoint: Dict,
                 logger: Dict,
                 **kwargs):
        self._device = torch.device('cpu')
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        self._total_epoch: int = total_epoch
        self._start_epoch: int = start_epoch
        self._alphabet: GraphAlphabet = GraphAlphabet(**alphabet)
        self._label: GraphLabel = GraphLabel(**label)
        self._model: LossModel = LossModel(**loss_model,
                                           device=self._device,
                                           vocab=self._alphabet.size(),
                                           class_num=self._label.size())
        cls = getattr(optim, optimizer['name'])
        self._optimizer: optim.Optimizer = cls(self._model.parameters(),
                                               **optimizer['params'])
        self._checkpoint: Checkpoint = Checkpoint(**checkpoint)
        self._accurate: Accurate = Accurate()
        self._logger: Logger = Logger(**logger)
        self._train: GraphDataset = GraphLoader(**train,
                                                label=self._label,
                                                alphabet=self._alphabet).build()
        # self._valid: GraphDataset = GraphLoader(**valid,
        #                                         label=self._label,
        #                                         alphabet=self._alphabet).build()
        self._test: GraphDataset = GraphLoader(**test,
                                               label=self._label,
                                               alphabet=self._alphabet).build()
        self._step: int = 0
        self._best: float = 0.0

    def train(self):
        self.load()
        self._logger.report_delimiter()
        self._logger.report_time("Starting:")
        self._logger.report_delimiter()
        for epoch in range(self._start_epoch, self._total_epoch + 1):
            self._logger.report_delimiter()
            self._logger.report_time("Epoch {}:".format(epoch))
            self._logger.report_delimiter()
            # self.train_step()
            train_rs = self.train_step()
            # valid_rs = self.valid_step()
            test_rs = self.test_step()
            self.save(train_rs, test_rs, epoch)
        self._logger.report_delimiter()
        self._logger.report_time("Finish:")
        self._logger.report_delimiter()

    def train_step(self):
        self._model.train()
        train_loss: Averager = Averager()
        for batch, (graphs,
                    labels,
                    texts,
                    lengths,
                    node_factors,
                    edge_factors,
                    node_sizes,
                    edge_sizes) in enumerate(self._train):
            self._optimizer.zero_grad()
            score, loss = self._model(graphs,
                                      labels,
                                      texts,
                                      lengths,
                                      node_factors,
                                      edge_factors,
                                      node_sizes,
                                      edge_sizes)
            loss.backward()
            self._optimizer.step()
            train_loss.update(loss.item() * labels.size(0), labels.size(0))
            self._step += 1
            # if self._step % 150 == 0:
            #     self._logger.report_delimiter()
            #     self._logger.report_time("Step {}:".format(self._step))
            #     self._logger.report_delimiter()
            #     valid_rs = self.valid_step()
            #     test_rs = self.test_step()
            #     self.save({
            #         "loss": train_loss.calc()
            #     }, valid_rs, test_rs, self._step)
            #     train_loss.clear()
            #     self._model.train()
        return {"loss": train_loss.calc()}

    # def valid_step(self):
    #     self._model.eval()
    #     valid_loss: Averager = Averager()
    #     all_score: List = []
    #     all_label: List = []
    #     with torch.no_grad():
    #         for batch, (graphs, labels, texts, lengths,
    #                     node_factors, edge_factors,
    #                     node_sizes, edge_sizes) in enumerate(self._valid):
    #             score, loss = self._model(graphs,
    #                                       labels,
    #                                       texts,
    #                                       lengths,
    #                                       node_factors,
    #                                       edge_factors,
    #                                       node_sizes,
    #                                       edge_sizes)
    #             valid_loss.update(loss.item(), labels.size(0))
    #             all_score.append(score)
    #             all_label.append(labels)
    #         metric, avg_f1 = self._gather(all_score, all_label)
    #     return {
    #         "loss": valid_loss.calc(),
    #         "avg_f1": avg_f1.calc(),
    #         "metric": metric
    #     }

    def test_step(self):
        self._model.eval()
        all_score: List = []
        all_label: List = []
        with torch.no_grad():
            for batch, (graphs, labels, texts, lengths,
                        node_factors, edge_factors,
                        node_sizes, edge_sizes) in enumerate(self._test):
                score = self._model(graphs,
                                    labels,
                                    texts,
                                    lengths,
                                    node_factors,
                                    edge_factors,
                                    node_sizes,
                                    edge_sizes,
                                    training=False)
                all_score.append(score)
                all_label.append(labels)
            metric, avg_f1 = self._gather(all_score, all_label)
        return {
            "avg_f1": avg_f1.calc(),
            "metric": metric
        }

    def _gather(self, score: List, label: List):
        avg_f1: Averager = Averager()
        recall, precision, f1_score = self._accurate(
            torch.cat(score, dim=0),
            torch.cat(label, dim=0),
            self._label.size()
        )
        metric: List = []
        recall = recall.tolist()
        precision = precision.tolist()
        f1_score = f1_score.tolist()
        for i in range(self._label.size()):
            metric.append({
                "name": self._label.decode(i),
                "recall": recall[i],
                "precision": precision[i],
                "f1_score": f1_score[i]
            })
            avg_f1.update(f1_score[i], 1)
        return metric, avg_f1

    def load(self):
        state_dict: Tuple = self._checkpoint.load()
        if state_dict is not None:
            self._model.load_state_dict(state_dict[0])
            self._optimizer.load_state_dict(state_dict[1])
            self._start_epoch = state_dict[2] + 1

    def save(self, train_rs: Dict, test_rs: Dict, epoch: int):
        self._logger.report_metric("training", train_rs)
        # self._logger.report_metric("validation", {
        #     "loss": valid_rs['loss'],
        #     "avg_f1": valid_rs['avg_f1']
        # })
        self._logger.report_metric("testing", {
            "avg_f1": test_rs['avg_f1']
        })
        self._logger.write({
            'training': train_rs,
            # 'validation': valid_rs,
            'testing': test_rs
        })
        self._checkpoint.save_last(epoch, self._model, self._optimizer)
        if test_rs['avg_f1'] > self._best:
            self._best = test_rs['avg_f1']
            self._checkpoint.save_model(self._model, epoch)
        self._logger.report_metric("best", {
            "avg_f1": self._best
        })


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser("Training config")
    parser.add_argument("-c", "--config_path", type=str, default='', help="config path")
    parser.add_argument("-p", "--root_path", type=str, default='', help="dataset root path")
    parser.add_argument("-a", "--alphabet_path", type=str, default='', help="alphabet path")
    parser.add_argument("-l", "--label_path", type=str, default='', help="label path")
    parser.add_argument("-r", "--resume", type=str, default='', help="checkpoint path")
    args = parser.parse_args()
    if args.config_path.strip():
        with open(args.config_path.strip()) as f:
            data: dict = yaml.safe_load(f)
    if args.alphabet_path.strip():
        data['alphabet']['alphabet_path'] = args.alphabet_path.strip()
    if args.label_path.strip():
        data['label']['label_path'] = args.label_path.strip()
    if args.resume.strip():
        data['checkpoint']['resume'] = args.resume.strip()
    if args.root_path.strip():
        tmp: str = args.root_path.strip()
        for item in ['train', 'test']:
            data[item]['dataset']['path'] = os.path.join(tmp, "{}.json".format(item))
    trainer = Trainer(**data)
    trainer.train()
