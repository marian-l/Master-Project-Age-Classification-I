from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForSequenceClassification
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

@dataclass
class DataCollatorSequenceWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch['labels'] = labels 

        return batch

def prepare_dataset(batch):
    audio = batch['path']
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch['labels'] = int(batch['age'])
    return batch

def compute_metrics_sklearn(y_pred, y_test):
  accuracy = accuracy_score(y_test,y_pred)
  return {"accuracy": accuracy}

def macro_averaged_mean_absolute_error(y_true, y_pred):
    c = np.unique(y_true)
    c_len = len(c)
    err = 0.0
    for i in c:
        idx = np.where(y_true == i)[0]
        y_true_label = np.take(y_true, idx)
        y_pred_label = np.take(y_pred, idx)
        err = err + mean_absolute_error(y_true=y_true_label, y_pred=y_pred_label)
    return err / c_len

def compute_metrics_3(pred):
    label_idx = [0,1,2,3,4,5,6,7]
    label_names = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties"]
    labels = pred.label_ids if len(pred.label_ids.shape) == 1 else pred.label_ids.argmax(-1)
    preds = pred.predictions if len(pred.predictions.shape) == 1 else pred.predictions.argmax(-1)
    print(f"labels: {labels}")
    print(f"preds: {preds}")
    acc = accuracy_score(labels, preds) 
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(y_true=labels, y_pred=preds, labels=label_idx, target_names=label_names)
    matrix = confusion_matrix(y_true=labels, y_pred=preds)
    print(report)
    print(matrix)
    mse = mean_squared_error(y_true=labels, y_pred=preds)
    mae = mean_absolute_error(y_true=labels, y_pred=preds)
    maem = macro_averaged_mean_absolute_error(y_true=labels, y_pred=preds)
    return {"accuracy": acc, "f1_score": f1, "MSE": mse, "MAE": mae, "MAE^M": maem}

training_args = TrainingArguments(
  output_dir="./wav2vec2-demo-mlmarian",
  group_by_length=False,
  per_device_train_batch_size=9,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
  load_best_model_at_end=True,
  metric_for_best_model="accuracy",
  greater_is_better=True,
  push_to_hub=True,
  hub_model_id = "mlmarian/wav2vec2-age-gender-balancedset",
  hub_token = "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",
  hub_private_repo=True,
)

# custom Wav2Vec2 Classifier-Model for 8 classes from Fivian and Reiser
class Wav2VecClassifierModelMean8(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)

        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 8)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        x = self.linear1(torch.mean(outputs[0], -2))
        x = self.tanh(x)
        x = self.linear2(x)

        result = {'logits': x}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  
            loss = loss_fct(x, labels)
            result['loss'] = loss 

        return result 


processor = Wav2Vec2Processor.from_pretrained('audeering/wav2vec2-large-robust-6-ft-age-gender')

AGES = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties"]

model = Wav2VecClassifierModelMean8.from_pretrained(
    "audeering/wav2vec2-large-robust-6-ft-age-gender",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    num_labels=8,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

common_voice_train = load_dataset('mlmarian/500x-per-age-class-filtered')
common_voice_train = common_voice_train.remove_columns(['client_id', 'gender', 'accents'])
common_voice_train.set_format("torch", device="cuda")
common_voice_train = common_voice_train.map(prepare_dataset, num_proc=1, batch_size=16)
common_voice_train = common_voice_train.remove_columns(['age', 'path'])
common_voice_train = common_voice_train['train'].train_test_split(test_size=0.2)
common_voice_eval = common_voice_train['test'].train_test_split(test_size=0.5)

data_collator = DataCollatorSequenceWithPadding(processor=processor, padding=True)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics_3,
    train_dataset=common_voice_train['train'],
    eval_dataset=common_voice_eval['train'],
)

train_result = trainer.train()

# model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"

trainer.evaluate(common_voice_eval['test'])