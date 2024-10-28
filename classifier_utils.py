import json
from IPython.core.display import HTML
from typing import Tuple, Any

import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import numpy as np

class Classifier:
    def __init__(self, model_path: str, id2cat_path: str, tokenizer_path: str = None):
        with open(id2cat_path, 'r') as handle:
            self.id2cat = json.load(handle)
        label_num = len(self.id2cat)
        if not tokenizer_path:
            tokenizer_path = model_path
        self.classifier = BertForSequenceClassification.from_pretrained(model_path, num_labels=label_num)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       model_max_length=512,
                                                       max_length=512,
                                                       truncation=True,
                                                       padding=False)


        self.input_string: str = ''

    def classifier_pass(self, string_sequence: str) -> Tuple[np.ndarray, Any, Any]:
        string_encoding = self.tokenizer(string_sequence,
                                         return_tensors='pt',
                                         return_token_type_ids=False,
                                         truncation=True,
                                         return_attention_mask=False)
        with torch.no_grad():
            classifier_output = self.classifier(**string_encoding, output_attentions=True)

        logits = classifier_output.logits.numpy()
        attentions = classifier_output.attentions[-1].numpy().squeeze()
        _, *attentions, _ = np.diag(np.average(attentions, axis=0)).tolist()
        tokens = self.tokenizer.tokenize(string_sequence)
        return logits, attentions, tokens

    @staticmethod
    def sigmoid(logits: np.ndarray) -> np.ndarray:
        return (np.exp(logits).T / np.exp(logits).sum(-1)).T

    @staticmethod
    def top_5(logits: np.ndarray) -> list:
        return np.argsort(logits.flatten())[::-1][:5].tolist()

    def __call__(self, string_sequence: str) -> Tuple[Any, Any, Any]:
        self.input_string = string_sequence
        logits, attentions, tokens = self.classifier_pass(string_sequence=self.input_string)
        predictions = self.top_5(logits=logits)
        return predictions, attentions, tokens


