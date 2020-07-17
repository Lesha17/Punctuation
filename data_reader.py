import transformers

import string

from nltk import wordpunct_tokenize
from torch.utils.data import Dataset
import os
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass
from typing import List, Tuple, Optional
from torch.utils.data.dataset import IterableDataset

from utils import PUNCT_TO_ID


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    labels: Optional[List[int]] = None


class PunctuationDataset(IterableDataset):
    def __init__(self, data_dir, tokenizer, label_to_idx, sentences_per_sample=3, max_len=192):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.label_to_idx = label_to_idx
        self.sentences_per_sample = sentences_per_sample
        self.max_length = max_len
        self._cached_len = None

    def __iter__(self):
        return iter(self.get_items())

    def __len__(self):
        if self._cached_len is None:
            print('Caclulating length of dataset', self.data_dir)
            self._cached_len = sum([1 for _ in iter(self)])
        return self._cached_len

    def get_items(self):
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(self.data_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath) as file:
                    sentences = sent_tokenize(file.read())
                    for i in range(len(sentences) - self.sentences_per_sample):
                        sentences = ' '.join(sentences[i:i + self.sentences_per_sample])
                        words, labels = self.process_text(sentences)
                        encoded = self.tokenizer(' '.join(words),
                                            max_length=self.max_length, padding="max_length", truncation=True)
                        label_idx = [self.label_to_idx[label] for label in labels]
                        if len(label_idx) > self.max_length:
                            label_idx = label_idx[:self.max_length]
                        else:
                            label_idx += [0] * (self.max_length - len(label_idx))
                        yield InputFeatures(**encoded, labels=label_idx)

    def process_text(self, text):
        tokens = wordpunct_tokenize(text)
        words = []
        labels = []
        for token in tokens:
            punct_symbols = [c for c in token if c in PUNCT_TO_ID]
            if len(punct_symbols) > 0: # Punctuation with 2 characters
                if len(labels) == 0:
                    print('Punctuation before any text:', text[:50], '...')
                    continue
                labels[-1] = punct_symbols[0] # Use the first punct symbol in this case
            else:
                words.append(token)
                labels.append(' ')
        return words, labels
