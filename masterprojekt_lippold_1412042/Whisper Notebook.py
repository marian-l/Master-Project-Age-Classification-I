from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, Trainer, TrainingArguments, WhisperForAudioClassification, TrainingArguments, Trainer
# from datasets import load_metric
from datasets import load_dataset
from datasets import Audio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_squared_error, mean_absolute_error

import json
import torchaudio
import torch
import torch.nn as nn
import torchvision
import numpy as np

def prepare_dataset(batch):
  audio = batch["path"]
  batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
  batch['labels'] = int(batch['age'])
  return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"labels": feature["labels"]} for feature in features]
        # label_features = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # labels = labels_batch["labels"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels_batch

        return batch

# classification head
class WhisperForAgeClassification(torch.nn.Module):
    def __init__(self, model, num_classes):
        super(WhisperForAgeClassification, self).__init__()
        self.whisper = model
        # self.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)  # Add a classification head
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 8)
        self.init_weights()

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None):
        outputs = self.whisper(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        # Use the last hidden state (or pool the hidden states) to classify the audio into age groups
        # hidden_states = outputs.last_hidden_state.mean(dim=1)  # Pooling (mean over time)
        # logits = self.classifier(hidden_states)  # Classification head
        # return logits

        x = self.linear1(torch.mean(outputs[0], -2))
        x = self.tanh(x)
        x = self.linear2(x)

        result = {'logits': x}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
            loss = loss_fct(x, labels)
            result['loss'] = loss  # Add loss to the output dictionary

        return result  # Return the logits (and loss if computed)







def compute_metrics(pred):
    # Get the predicted class IDs and true label IDs
    pred_ids = pred.predictions.argmax(-1)  # Assuming predictions are logits and you need the highest score
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id (if you're handling padding tokens)
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Compute accuracy
    accuracy = accuracy_score(label_ids, pred_ids)

    label_idx = [0,1,2,3,4,5,6,7]
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

#training_args = Seq2SeqTrainingArguments(
training_args = TrainingArguments(
    output_dir="./whisper-small-mlmarian",  # change to a repo name of your choice
    per_device_train_batch_size=9,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    #gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    #predict_with_generate=True,
    #generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    push_to_hub=True,
    hub_model_id = "mlmarian/whisper-finetuning",
    hub_token = "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",
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