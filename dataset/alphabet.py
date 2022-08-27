from typing import List, Dict
import json
import numpy as np
from utils import remove_space
import unicodedata


class GraphAlphabet:
    def __init__(self, alphabet_path: str) -> None:
        self.encoded_pad: int = 0
        self.decoded_pad: str = '<pad>'
        with open(alphabet_path, 'r', encoding='utf-8') as f:
            alphabet = remove_space(f.readline())
            alphabet = ' ' + alphabet
            alphabet = unicodedata.normalize("NFC", alphabet)
        self._character: Dict = {c: i + 1 for i, c in enumerate(alphabet)}
        self._character[self.decoded_pad] = self.encoded_pad
        self._number: Dict = {i + 1: c for i, c in enumerate(alphabet)}
        self._number[self.encoded_pad] = self.decoded_pad

    def encode(self, txt: str) -> np.ndarray:
        # convert character to number
        txt = unicodedata.normalize("NFC", txt)
        encoded_txt: List = [self._character.get(char, self.encoded_pad)
                             for char in txt.lower()]
        return np.array(encoded_txt, dtype=np.int32)

    def decode(self, encoded_txt: np.ndarray) -> str:
        # convert number to character
        decoded_txt: List = [self._number.get(num, self.decoded_pad) for num in encoded_txt]
        # remove padding character
        decoded_txt = [char for char in decoded_txt if char != self.decoded_pad]
        return ''.join(decoded_txt)

    def size(self):
        return len(self._number)


class GraphLabel:
    def __init__(self, label_path: str):
        self.encoded_other = 0
        self.decoded_other = "OTHER"
        with open(label_path, 'r', encoding='utf-8') as f:
            labels: List = json.loads(remove_space("".join(f.readlines())))
        # self._character: Dict = {label: i + 1 for i, label in enumerate(self._select_label)}
        self._character: Dict = {label.upper(): i + 1 for i, label in enumerate(labels)}
        self._character[self.decoded_other] = self.encoded_other
        # self._number: Dict = {i + 1: label for i, label in enumerate(self._select_label)}
        self._number: Dict = {i + 1: label.upper() for i, label in enumerate(labels)}
        self._number[self.encoded_other] = self.decoded_other

    def encode(self, label: str):
        # convert label to number
        return self._character.get(label.upper(), self.encoded_other)

    def decode(self, num: int):
        # convert number to label
        return self._number.get(num, self.decoded_other)

    def size(self):
        return len(self._character)
