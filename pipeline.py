import data_reader
import features


from typing import List, Tuple, Iterable, Set, Dict
import json
import os
import shutil
import transformers
#from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModel, AutoConfig
#from transformers.modeling_outputs import MultipleChoiceModelOutput
#from transformers import AdamW, get_scheduler
import datasets
import evaluate
#from datasets import load_dataset, Dataset, DatasetDict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


MODEL_NAME = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch: datasets.formatting.formatting.LazyBatch) -> transformers.tokenization_utils_base.BatchEncoding:
    return tokenizer(batch["segment"], truncation=True)  # max_length=512


class CustomModel(nn.Module):
    def __init__(self, model_name: str, n_levels: int) -> None:
        super(CustomModel, self).__init__()
        self.n_levels = n_levels

        # Load Model with given checkpoint and extract its body
        self.model = transformers.AutoModel.from_pretrained(
            model_name,
            config=transformers.AutoConfig.from_pretrained(
                model_name,
                output_attentions=True,
                output_hidden_states=True
            )
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, self.n_levels)  # load and initialize weights

    def load_state(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def save_state(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None) -> None:
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_levels), labels.view(-1))

        return transformers.modeling_outputs.MultipleChoiceModelOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


class Pipeline(object):
    def __init__(self, data_file: str, output_dir: str):
        self.data_file = data_file
        self.output_dir = output_dir
        self.full_dataset = None
        self.n_levels = None
        self.model = None
        self.dataset_file = None
        self.model_state_path = os.path.join(output_dir, "rf_model")
        self.dataset_file = os.path.join(self.output_dir, "all_data.json")
        self.datasets_cache_dir = os.path.join(self.output_dir, "datasets_cache")
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> None:
        all_data = self.preprocess_data()
        self.load_data()

        self.n_levels = max(all_data, key=lambda x: x["score"])["score"] + 1
        self.model = CustomModel(MODEL_NAME, n_levels=self.n_levels)
        try:
            self.model.load_state(self.model_state_path)
            print("Loaded model state from file")
        except FileNotFoundError:
            self.model_fit()
            self.model.save_state(self.model_state_path)
        #output_file = os.path.join(self.output_dir, "cv_results.txt")
        #result_df.to_csv(output_file, sep="\t", index=False)

    def model_fit(self) -> None:
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        tokenized_dataset = self.full_dataset.map(tokenize, batched=True)
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

        train_dataloader = DataLoader(
            tokenized_dataset["train"], shuffle=True, batch_size=32, collate_fn=data_collator
        )
        eval_dataloader = DataLoader(
            tokenized_dataset["test"], batch_size=32, collate_fn=data_collator
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)  # lr=5e-5

        num_epochs = 3
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = transformers.get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        print(num_training_steps)

        metric = evaluate.load("accuracy")

        progress_bar_train = tqdm(range(num_training_steps))
        progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar_train.update(1)

            self.model.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
                progress_bar_eval.update(1)

            print(metric.compute())

        self.model.eval()

    def preprocess_data(self) -> Dict:
        full_data = data_reader.read_full_dataset(self.data_file)
        all_data = features.create_features(full_data, min_score=full_data["RF-Score"].min())
        with open(self.dataset_file, "w", encoding="utf8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        return all_data

    def load_data(self) -> datasets.DatasetDict:
        # clear cache dir to avoid reading old data
        if os.path.exists(self.datasets_cache_dir):
            shutil.rmtree(self.datasets_cache_dir)

        data = datasets.load_dataset("json", data_files=self.dataset_file, cache_dir=self.datasets_cache_dir)
        data = data.rename_column("score", "label")

        data.set_format('pandas')
        data = data['train'][:]

        data.drop_duplicates(subset=['segment'], inplace=True)
        data = data.reset_index()[['segment', 'label']]
        data = datasets.Dataset.from_pandas(data)

        # 80% train, 20% test + validation
        train_testvalid = data.train_test_split(test_size=0.2, seed=15)

        # gather everyone if you want to have a single DatasetDict
        data = datasets.DatasetDict({
            'train': train_testvalid['train'],
            'test': train_testvalid['test']})
        self.full_dataset = data
        return data


if __name__ == '__main__':
    pipeline = Pipeline(r"..\data\Beispiel-Segmente CRF.xlsx", output_dir=r"..\output")
    pipeline.run()
    #tokenizer.write_vocabulary_by_frequency(pipeline.vocabulary, handle=sys.stdout, sorting="lexicographic")
    from pprint import pprint
    #pprint(tokenizer.tokens_to_sentences())
