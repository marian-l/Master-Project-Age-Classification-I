from datasets import load_dataset, ClassLabel
from transformers import AutoFeatureExtractor
from datasets import Audio

from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, mean_squared_error, mean_absolute_error

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

dataset = load_dataset("mlmarian/500x-per-age-class-filtered")
dataset = dataset.select_columns(["path", "age"])
labels = dataset["train"].features["age"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

model_id = "openai/whisper-small"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True
)

sampling_rate = feature_extractor.sampling_rate

dataset = dataset.cast_column("path", Audio(sampling_rate=sampling_rate))
max_duration = 30

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["path"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
    )
    return inputs

def preprocess_function_2(batch):
  batch['labels'] = batch['age']
  return batch

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def macro_averaged_mean_absolute_error(y_true, y_pred):
    unique_classes = np.unique(y_true)
    class_count = len(unique_classes)
    error_sum = 0.0

    for class_label in unique_classes:
        class_indices = np.where(y_true == class_label)[0]
        y_true_class = np.take(y_true, class_indices)
        y_pred_class = np.take(y_pred, class_indices)
        error_sum += mean_absolute_error(y_true=y_true_class, y_pred=y_pred_class)

    return error_sum / class_count

def compute_metrics_2(eval_pred):
    """Computes accuracy, F1 score, confusion matrix, and more on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    accuracy = accuracy_score(labels, predictions)
    label_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    label_names = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties"]

    f1 = f1_score(labels, predictions, average='macro')
    report = classification_report(labels, predictions, labels=label_idx, target_names=label_names)
    matrix = confusion_matrix(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    maem = macro_averaged_mean_absolute_error(labels, predictions)

    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{matrix}")
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "MSE": mse,
        "MAE": mae,
        "MAE^M": maem,
    }

dataset_encoded = dataset.map(
    preprocess_function,
    remove_columns='path',
    batched=True,
    batch_size=2,
    num_proc=1,
)

dataset_encoded = dataset_encoded.map(
    preprocess_function_2,
    remove_columns='age',
    batched=True,
    batch_size=2,
    num_proc=1,
)

dataset_encoded = dataset_encoded['train'].train_test_split(test_size=0.2)
eval_dataset = dataset_encoded['test'].train_test_split(test_size=0.5)

accuracy = evaluate.load("accuracy")

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
)

# Remove invalid keys from model config (should be in a GenerationConfig instead.)
generation_keys = ['max_length', 'suppress_tokens', 'begin_suppress_tokens']
for key in generation_keys:
    if hasattr(model.config, key):
        delattr(model.config, key)  # Remove the attribute from the config

model_name = model_id.split("/")[-1]
batch_size = 2
gradient_accumulation_steps = 1
num_train_epochs = 10 # 10 seems the minimum for more than 5 classes

training_args = TrainingArguments(
    output_dir="./test_whisper",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=True,
    hub_model_id = "mlmarian/whisper-finetuning",
    hub_private_repo=True,
)

metric = evaluate.load("accuracy")

trainer = Trainer(
    model,
    training_args,
    train_dataset=dataset_encoded["train"].with_format("torch"),
    eval_dataset=eval_dataset["train"].with_format("torch"),
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics_2,
)

trainer.train()

trainer.evaluate(eval_dataset['test'])