import pandas
from datasets import Dataset, DatasetDict
import torch
import torchaudio
import torch_directml
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, TrainingArguments, Trainer

# print(torch.__version__)
# print(torchaudio.__version__)

audio_df = pandas.read_csv(filepath_or_buffer='../_datasets/CommonVoice/dev-train-set/male-speakers-clips.csv',
                           header=0, encoding='utf-8')
audio_df = audio_df[['path', 'age']]

audio_file_paths = 'D:\Sprachdaten\cv-corpus-17.0-2024-03-15-de\cv-corpus-17.0-2024-03-15\de\clips\\' + audio_df['path']
age_labels = audio_df['age']


def load_audio(batch):
    waveform, sample_rate = torchaudio.load(batch['file'])
    batch['audio'] = {'array': waveform.numpy(), 'sampling_rate': sample_rate}
    return batch


data = {'file': audio_file_paths, 'label': age_labels}
dataset = Dataset.from_dict(data)

dataset = dataset.map(load_audio)

# Split the dataset into train, validation, and test sets
dataset = dataset.train_test_split(test_size=0.2)
train_testvalid = dataset['train'].train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_testvalid['train'],
    'validation': train_testvalid['test'],
    'test': dataset['test']
})

# Load the processor and model
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base', num_labels=1)

# Preprocess function
def preprocess_function(batch):
    inputs = processor(batch['audio']['array'], sampling_rate=batch['audio']['sampling_rate'], return_tensors='pt')
    batch['input_values'] = inputs.input_values[0]
    batch['labels'] = torch.tensor(batch['label'], dtype=torch.float32)
    return batch

# Apply preprocessing
encoded_dataset = dataset.map(preprocess_function, remove_columns=['file', 'audio'])

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=processor.feature_extractor,
)

# Train the model
trainer.train()

results = trainer.evaluate(encoded_dataset['test'])
print(results)

model.push_to_hub("mlmarian/wav2vec2base-CV-HRSM24")
processor.push_to_hub("mlmarian/wav2vec2base-CV-HRSM24")