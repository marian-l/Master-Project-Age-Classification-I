from datasets import Audio
from datasets import load_dataset
import json

import torchaudio

import torch
import torchvision
import numpy as np
# import wandb

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import TrainingArguments
from transformers import Trainer
from transformers import Wav2Vec2Processor

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

vocab_dict = {' ': 1,
              "'": 2,
              'a': 3,
              'b': 4,
              'c': 5,
              'd': 6,
              'e': 7,
              'f': 8,
              'g': 9,
              'h': 10,
              'i': 11,
              'j': 12,
              'k': 13,
              'l': 14,
              'm': 15,
              'n': 16,
              'o': 17,
              'p': 18,
              'q': 19,
              'r': 20,
              's': 21,
              't': 22,
              'u': 23,
              'v': 24,
              'w': 25,
              'x': 26,
              'y': 27,
              'z': 28,
              'ä': 29,
              'ü': 30,
              'ö': 31,
              'ß': 32,
              '̇' : 33}
# The model has to learn to predict when a word is finished or else the model prediction would always be a sequence of chars which would make it impossible to separate words from each other.


# we add a more clear definition of " "
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# we add unknown characters
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

# save the vocabulary

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# create Tokenizer
from transformers import Wav2Vec2CTCTokenizer
# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# One should always keep in mind that the data-preprocessing is a very important step before training your model. E.g., we don't want our model to differentiate between a and A just because we forgot to normalize the data. The difference between a and A does not depend on the "sound" of the letter at all, but more on grammatical rules - e.g. use a capitalized letter at the beginning of the sentence. So it is sensible to remove the difference between capitalized and non-capitalized letters so that the model has an easier time learning to transcribe speech.

# create FeatureExtractor
from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, # raw speech signal
                                             sampling_rate=16000,
                                             padding_value=0.0, # shorter inputs need to be padded
                                             do_normalize=True, # whether the input should be zero-mean-unit-variance normalized or not. usually better performance with True
                                             return_attention_mask=True) # generally, yes

# create Processor
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

AGES = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties"]

def prepare_dataset(batch):
    audio = batch['path']

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch['labels'] = int(batch['age'])
        # batch["labels"] = (batch["age"])  # Ensure 'age' is mapped to an integer label
        # batch["labels"] = processor(str(batch["age"])).input_ids # for ASR-tasks
        # batch["labels"] = processor(int(batch["age"]))  # Ensure 'age' is mapped to an integer label
        # batch["labels"] = processor(int(batch["age"])).input_ids  # Ensure 'age' is mapped to an integer label
    return batch

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

    # encoder = OneHotEncoder(sparse=False)
    # onehot_labels = encoder.fit_transform(labels.reshape(-1, 1))

    acc = accuracy_score(labels, preds) # error: got 0 for y_true. maybe onehot the preds here?
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(y_true=labels, y_pred=preds, labels=label_idx, target_names=label_names)
    matrix = confusion_matrix(y_true=labels, y_pred=preds)
    print(report)
    print(matrix)
    # wandb.log(
    #     {"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds, class_names=label_names)})
    # wandb.log(
    #     {"precision_recall": wandb.plot.pr_curve(y_true=labels, y_probas=pred.predictions, labels=label_names)})
    mse = mean_squared_error(y_true=labels, y_pred=preds)
    mae = mean_absolute_error(y_true=labels, y_pred=preds)
    maem = macro_averaged_mean_absolute_error(y_true=labels, y_pred=preds)
    return {"accuracy": acc, "f1_score": f1, "MSE": mse, "MAE": mae, "MAE^M": maem}

from transformers import Wav2Vec2ForSequenceClassification
#################################################################################
# load, slice and prepare dataset
# common_voice_train = load_dataset('mlmarian/MasterProjectPart3')
common_voice_train = load_dataset('mlmarian/sample_set_size_1')

common_voice_train = common_voice_train.remove_columns(['client_id', 'gender', 'accents'])

common_voice_train.set_format("torch", device="cuda")

common_voice_train = common_voice_train.map(prepare_dataset, num_proc=1, batch_size=16)
# common_voice_train = common_voice_train.remove_columns(['age', 'path'])

common_voice_train = common_voice_train['train'].train_test_split(test_size=0.2)

# test steps
data_collator = DataCollatorSequenceWithPadding(processor=processor, padding=True)
accuracy_metric = load_metric("accuracy", trust_remote_code=True)

# MODEL
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
    # "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    num_labels=8,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

model.freeze_feature_extractor() # The first component of XLS-R consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretraining and as stated in the paper does not need to be fine-tuned anymore. Thus, we can set the requires_grad to False for all parameters of the feature extraction part. https://github.com/huggingface/blog/blob/main/fine-tune-xlsr-wav2vec2.md
model.gradient_checkpointing_enable()

# EXPERIMENTAL REGRESSION PROBLEM
# def compute_metrics_regression(pred):
#     predictions = pred.predictions
#     labels = pred.label_ids
#     mse = ((predictions - labels) ** 2).mean()
#     return {"mse": mse}
#
# model.config.problem_type = "regression"
###############################

# GPU nutzen!
# Batch-Size anpassen
# Datensatz ggf. REPRÄSENTATIV verkleinern
# Paper: Parameter nur dokumentieren, wenn nicht standart oder verändert

# TRAINING
training_args = TrainingArguments(
  output_dir="./wav2vec2-xlsr-demo-mlmarian",
  group_by_length=False,
  # per_device_train_batch_size=4,
  per_device_train_batch_size=9,
  # per_device_train_batch_size=16,
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
  hub_model_id = "mlmarian/wav2vec2-xlrs-finetuning",
  hub_token = "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",
  hub_private_repo=True,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics_3,
    train_dataset=common_voice_train['train'],
    eval_dataset=common_voice_train['test'],
)

trainer.train()
#huggingface pytorch process data 2 gpu

from huggingface_hub import HfApi
api = HfApi(token="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")
api.get_token_permission(token="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")

trainer.push_to_hub("mlmarian/wav2vec2-xlrs-finetuning")
trainer.save_model("mlmarian/wav2vec2-xlrs-finetuning")


# not used
def compute_metrics(pred):
  pred_logits = pred.predictions
  pred_ids = np.argmax(pred_logits, axis=-1)
  pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
  pred_str = processor.batch_decode(pred_ids)
  # we do not want to group tokens when computing the metrics
  label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
  accuracy = accuracy_metric.compute(predictions=pred_str, references=label_str)
  return {"accuracy": accuracy_metric}

def compute_metrics_2(pred):
  pred_logits = pred.predictions
  pred_ids = np.argmax(pred_logits, axis=-1)
  # Replace `-100` in label_ids (if applicable)
  label_ids = pred.label_ids
  label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
  # Compute accuracy at the token/label level
  accuracy = (pred_ids == label_ids).mean()

  return {"accuracy": accuracy}

def compute_metrics_sklearn(y_pred, y_test):
  accuracy = accuracy_score(y_test,y_pred)
  return {"accuracy": accuracy}

