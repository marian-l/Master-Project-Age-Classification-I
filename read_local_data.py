from datasets import load_dataset, Audio, Dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16000))

print(minds[0])

audio_dataset = Dataset.from_dict({"audio": [""]})