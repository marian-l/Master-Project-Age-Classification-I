def load_audio(batch):
    waveform, sample_rate = torchaudio.load(batch['file'])
    batch['audio'] = {'array': waveform.numpy(), 'sampling_rate': sample_rate}
    return batch

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