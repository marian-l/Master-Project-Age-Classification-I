from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Trainer, TrainingArguments, \
    WhisperForAudioClassification, TrainingArguments, Trainer
# from datasets import load_metric
from datasets import load_dataset
from datasets import Audio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_squared_error, \
    mean_absolute_error

import json
import torchaudio
import torch
import torchvision
import numpy as np


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["path"]

    # compute log-Mel input features from input audio array
    # batch["input_features"] = feature_extractor(audio["array"].cpu(), sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # batch["input_features"] = feature_extractor(audio["array"].cpu(), sampling_rate=audio["sampling_rate"]).input_features[0]
    # batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    # print(batch['age'])
    # batch["labels"] = tokenizer(batch["sentence"]).input_ids
    batch['labels'] = int(batch['age'].item())  # print(batch['labels']) works correctly
    # batch['labels'] = int(batch['age'].cpu()) #  print(batch['labels']) works correctly

    # with processor.as_target_processor():
    #   batch['labels'] = int(batch['age'])
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]

        #TODO print(features[0].keys()) # batch['labels'] does not exist?

        # label_features = [{"labels": feature["labels"]} for feature in features]
        label_features = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["labels"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        batch["labels"] = labels

        return batch


# classification head
class WhisperForAgeClassification(torch.nn.Module):
    def __init__(self, model, num_classes):
        super(WhisperForAgeClassification, self).__init__()
        self.whisper = model
        self.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)  # Add a classification head

    def forward(self, input_features):
        outputs = self.whisper(input_features)
        hidden_states = outputs.last_hidden_state.mean(dim=1)  # Pooling (mean over time)
        logits = self.classifier(hidden_states)  # Classification head
        return logits


def compute_metrics(pred):
    pred_ids = pred.predictions.argmax(-1)  # Assuming predictions are logits and you need the highest score
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    accuracy = accuracy_score(label_ids, pred_ids)

    label_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    label_names = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties"]
    labels = pred.label_ids if len(pred.label_ids.shape) == 1 else pred.label_ids.argmax(-1)
    preds = pred.predictions if len(pred.predictions.shape) == 1 else pred.predictions.argmax(-1)
    print(f"labels: {labels}")
    print(f"preds: {preds}")
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(y_true=labels, y_pred=preds, labels=label_idx, target_names=label_names)
    matrix = confusion_matrix(y_true=labels, y_pred=preds)
    print(report)
    print(matrix)

    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./whisper-small-mlmarian",  # change to a repo name of your choice
    per_device_train_batch_size=9,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    push_to_hub=True,
    hub_model_id="mlmarian/whisper-finetuning",
    hub_token="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",
    hub_private_repo=True,
)

model = WhisperForAudioClassification.from_pretrained(
    "openai/whisper-small",
    attention_dropout=0.1,
    mask_time_prob=0.05,
    num_labels=8)

num_classes = 8

classification_model = WhisperForAgeClassification(model, num_classes)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="german")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="german")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

common_voice_train = load_dataset('mlmarian/sample_set_size_1')
common_voice_train = common_voice_train.remove_columns(['client_id', 'gender', 'accents'])

if torch.cuda.is_available():
    common_voice_train.set_format("torch", device="cuda")

common_voice_train = common_voice_train.map(prepare_dataset, num_proc=1, batch_size=16)

common_voice_train = common_voice_train.remove_columns(['age', 'path'])
common_voice_train = common_voice_train['train'].train_test_split(test_size=0.2)

trainer = Trainer(
    args=training_args,
    model=classification_model,
    train_dataset=common_voice_train["train"],
    eval_dataset=common_voice_train["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()