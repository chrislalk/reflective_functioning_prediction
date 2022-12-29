import data_reader
import features


from typing import List, Tuple, Iterable, Set
import pandas as pd
import numpy as np
import json
import os
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers import AdamW, get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class CustomModel(nn.Module):
    def __init__(self, model_name, n_levels):
        super(CustomModel, self).__init__()
        self.n_levels = n_levels

        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(model_name,
                                               config=AutoConfig.from_pretrained(model_name,
                                                                                 output_attentions=True,
                                                                                 output_hidden_states=True))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.n_levels)  # load and initialize weights

    def load_state(self, path):
        self.load_state_dict(torch.load(path))

    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_levels), labels.view(-1))

        return MultipleChoiceModelOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                         attentions=outputs.attentions)


class Pipeline(object):
    def __init__(self, data_file, output_dir):
        self.data_file = data_file
        self.output_dir = output_dir
        self.full_dataset = None
        self.n_levels = None
        self.model = None
        self.model_state_path = os.path.join(output_dir, "rf_model")
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> None:
        full_data = data_reader.read_full_dataset(self.data_file)
        all_data = features.create_features(full_data, min_score=full_data["RF-Score"].min())
        with open(os.path.join(self.output_dir, "all_data.json"), "w", encoding="utf8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        print(all_data)
        self.n_levels = max(all_data, key=lambda x: x["score"])["score"]
        self.model = CustomModel(MODEL_NAME, n_levels=self.n_levels)
        try:
            print("Loaded model state from file")
            self.model.load_state(self.model_state_path)
        except FileNotFoundError:
            pass
        #output_file = os.path.join(self.output_dir, "cv_results.txt")
        #result_df.to_csv(output_file, sep="\t", index=False)


if __name__ == '__main__':
    pipeline = Pipeline(r"..\data\Beispiel-Segmente CRF.xlsx", output_dir=r"..\output")
    pipeline.run()
    #tokenizer.write_vocabulary_by_frequency(pipeline.vocabulary, handle=sys.stdout, sorting="lexicographic")
    from pprint import pprint
    #pprint(tokenizer.tokens_to_sentences())
