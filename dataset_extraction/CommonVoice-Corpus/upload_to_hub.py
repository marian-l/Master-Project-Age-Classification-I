# from datasets import load_from_disk, Dataset, load_dataset, concatenate_datasets
from transformers import Trainer, WhisperForAudioClassification
# import datasets
from huggingface_hub import HfApi
import os

api = HfApi(token="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")
# api = HfApi(token="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")
test = api.get_token_permission(token="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")
repo_name = "mlmarian/commonvoice_masterproject_test3"

# dataset = load_from_disk("../_datasets/CommonVoice/dev-train-set/full_dataset.arrow")

# dataset.push_to_hub(repo_name)
trainer = Trainer()
trainer.save_model()

# Function to load and concatenate datasets
def concatenate_arrow_files(file_list):
    datasets = [Dataset.from_file(f) for f in file_list]
    concatenated_dataset = concatenate_datasets(*datasets)
    return concatenated_dataset

data_dir = "../_datasets/CommonVoice/dev-train-set/full_dataset.arrow"

arrow_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.arrow')]

# Iterate over the .arrow files and concatenate them in groups of 3
batch_size = 3
for i in range(0, len(arrow_files), batch_size):
    batch_files = arrow_files[i:i + batch_size]
    concatenated_dataset = concatenate_arrow_files(batch_files)

    # Define the name of the concatenated file
    output_file = f"concatenated_part_{i // batch_size + 1}.arrow"

    destination = f'../_datasets/CommonVoice/dev-train-set/"{output_file}'
    concatenated_dataset.save_to_disk(destination)

    concatenated_dataset.push_to_hub(repo_name + "_" + str(i))

# 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


