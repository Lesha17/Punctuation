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
                        encoded, label_idx = self.process_text(sentences)
                        yield InputFeatures(**encoded, labels=label_idx)

    def pad_to_max_len(self, seq):
        return seq + [0] * (self.max_length - len(seq))

    def process_text(self, text):
        tokenized = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True)
        result_token_ids = []
        token_type_ids = []
        attention_mask = []
        labels = []
        for idx, tti, att_msk in list(zip(tokenized.input_ids, tokenized.token_type_ids, tokenized.attention_mask)):
            token = self.tokenizer.ids_to_tokens[idx]
            if token in PUNCT_TO_ID:
                if len(labels) == 0:
                    print('Punctuation before any text:', text[:50], '...')
                    continue
                labels[-1] = token # Use the first punct symbol in this case
            else:
                result_token_ids.append(idx)
                token_type_ids.append(tti)
                attention_mask.append(att_msk)
                labels.append(' ')


        result_encoded = {
            'input_ids': self.pad_to_max_len(result_token_ids),
            'token_type_ids': self.pad_to_max_len(token_type_ids),
            'attention_mask': self.pad_to_max_len(attention_mask)
        }
        return result_encoded, self.pad_to_max_len([self.label_to_idx[label] for label in labels])
