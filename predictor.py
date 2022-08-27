import copy
import json
import math
import random
import torch
import numpy as np
import dgl
import yaml
from typing import Dict, List
from loss_model import LossModel
from dataset import GraphAlphabet, GraphLabel, norm
from torch import Tensor
import warnings
import argparse
import time
import cv2 as cv


class BREGPredictor:
    def __init__(self, config: str, alphabet: str, label: str, pretrained: str):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        with open(config) as f:
            data: Dict = yaml.safe_load(f)
        self.alphabet: GraphAlphabet = GraphAlphabet(alphabet_path=alphabet)
        self.label: GraphLabel = GraphLabel(label_path=label)
        self.model = LossModel(vocab=self.alphabet.size(),
                               class_num=self.label.size(),
                               device=self.device,
                               **data['loss_model'])
        state_dict = torch.load(pretrained, map_location=self.device)
        self.model.load_state_dict(state_dict['model'])

    def _process_ocr(self, ocr_result: Dict):
        TARGET_KEY = "target"
        TEXT_KEY = "text"
        LABEL_KEY = "label"
        BBOX_KEY = "box"
        SHAPE_KEY = "shape"

        lengths = []
        texts = []
        bboxes = []
        labels = []
        for target in ocr_result[TARGET_KEY]:
            text = self.alphabet.encode(target[TEXT_KEY])
            if text.shape[0] == 0:
                continue
            texts.append(text)
            lengths.append(text.shape[0])
            label: int = self.label.encode(target[LABEL_KEY])
            labels.append(label)
            (x, y), (w, h), a = cv.minAreaRect(np.array(target[BBOX_KEY]).astype(np.int32))
            if w >= h:
                bbox = np.array([x, y, w, h])
            else:
                bbox = np.array([y, x, h, w])
            bboxes.append(bbox)
        return (np.array(bboxes),
                np.array(labels),
                np.array(texts),
                np.array(lengths))

    def _preprocess(self, ocr_result: Dict):
        bboxes, labels, texts, lengths = self._process_ocr(ocr_result)
        node_size = labels.shape[0]
        src: List = []
        dst: List = []
        dists: List = []
        for i in range(node_size):
            x_i, y_i, w_i, h_i = bboxes[i]
            for j in range(node_size):
                if i == j:
                    continue

                x_j, y_j, w_j, h_j = bboxes[j]

                x_dist = x_j - x_i
                y_dist = y_j - y_i

                # if abs(y_dist) > 3 * h_i:
                #     continue
                dists.append([abs(x_dist), abs(y_dist),
                              lengths[j] / lengths[i]])
                src.append(i)
                dst.append(j)
        print("edge num", len(src))
        print("node num", node_size)
        g = dgl.DGLGraph()
        g.add_nodes(node_size)
        g.add_edges(src, dst)
        g.ndata['feat'] = torch.FloatTensor(norm(bboxes))
        g.edata['feat'] = torch.FloatTensor(norm(np.array(dists)))

        node_nums = g.number_of_nodes()
        node_factor = torch.FloatTensor(node_nums, 1).fill_(1. / float(node_nums))
        node_factor = node_factor.sqrt()

        edge_nums = g.number_of_edges()
        edge_factor = torch.FloatTensor(edge_nums, 1).fill_(1. / float(edge_nums))
        edge_factor = edge_factor.sqrt()

        max_length = np.max(lengths)
        new_text = [np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), 'constant'), axis=0)
                    for t in texts]
        texts = np.concatenate(new_text)

        texts = torch.from_numpy(np.array(texts)).long()
        lengths = torch.from_numpy(np.array(lengths)).long()

        return (g,
                texts,
                lengths,
                node_factor,
                edge_factor,
                node_nums,
                edge_nums)

    def predict(self, ocr_result: Dict):
        input_data = self._preprocess(ocr_result)
        score: Tensor = self.model.predict(*input_data)
        values, pred = torch.log_softmax(score.cpu(), dim=1).max(1)
        result = [(self.label.decode(pred[i].item()), values[i].item())
                  for i in range(len(pred))]
        pred_ocr = copy.deepcopy(ocr_result)
        for ocr, label in zip(pred_ocr["target"], result):
            ocr["label"] = label[0]
            ocr["label_score"] = math.exp(label[1])
        return pred_ocr


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser("Predictor config")
    parser.add_argument("-c", "--config_path", type=str, default='', help="config path")
    parser.add_argument("-r", "--resume", type=str, default='', help="checkpoint path")
    parser.add_argument("-a", "--alphabet_path", type=str, default='', help="alphabet path")
    parser.add_argument("-l", "--label_path", type=str, default='', help="label path")
    parser.add_argument("-i", "--input", type=str, default='', help="input data_v1 path")
    args = parser.parse_args()
    predictor = BREGPredictor(args.config_path.strip(),
                              args.alphabet_path.strip(),
                              args.label_path.strip(),
                              args.resume.strip())
    with open(args.input.strip(), 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    start = time.time()
    # id = random.randint(0, len(data_v1) - 1)
    id = random.randint(0, len(data) - 1)
    output = predictor.predict(data[id])
    print("Run_time:", time.time() - start)
    wrong: int = 0
    with open(r"D:\python_project\breg_graph\result\abc_{}.json".format(id), 'w', encoding='utf-8') as f:
        f.write(data[id]['file_name'])
        f.write('\n')
        for pred, gt in zip(output['target'], data[id]['target']):
            print("-" * 50)
            f.write("-" * 50)
            f.write("\n")
            print("Pred:", pred["text"], pred['label'].upper(), pred['label_score'])
            f.write("Pred: {} {} {}".format(pred["text"], pred['label'].upper(), pred['label_score']))
            f.write("\n")
            print("GT:", gt["text"], gt['label'].upper())
            f.write("GT: {}".format(gt["text"]))
            f.write("\n")
            if pred['label'].upper() != gt['label'].upper():
                wrong += 1
            print("-" * 50)
            f.write("-" * 50)
            f.write("\n")
        print("Wrong", wrong)
        print("Total", len(data[id]['target']))
