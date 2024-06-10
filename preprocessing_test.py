from transformers import WhisperFeatureExtractor
from datasets import Audio, load_dataset
import librosa
import numpy as np
import matplotlib.pyplot as plt


def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS


def prepare_dataset_for_whisper(example):
    audio = example["audio"]
    features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], padding=True)
    return features


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

MAX_DURATION_IN_SECONDS = 30.0

# use librosa to get examples duration from the audio file
new_column = [librosa.get_duration(path=x) for x in minds['path']]
minds = minds.add_column("duration", new_column)

# use 🤗 Datasets' `filter` method to apply the filtering function
minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])

# remove the temporary helper column
minds = minds.remove_columns(["duration"])
print(minds)

minds = minds.map(prepare_dataset_for_whisper)

example = minds[0]
input_features = example["input_features"]

plt.figure().set_figwidth(12)
librosa.display.specshow(
    np.asarray(input_features[0]),
    x_axis="time",
    y_axis="mel",
    sr=feature_extractor.sampling_rate,
    hop_length=feature_extractor.hop_length,
)
plt.colorbar()
plt.show()