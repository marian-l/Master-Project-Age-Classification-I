import pandas as pd
import numpy as np

PATH = "D:/Sprachdaten/cv-corpus-17.0-2024-03-15-de/cv-corpus-17.0-2024-03-15/de/"
VALIDATED = "validated.tsv"
CLIP_DURATIONS = "clip_durations.tsv"
OTHERS = "other - Kopie.tsv"

path_to_validated = PATH + VALIDATED
path_to_others = PATH + OTHERS

AGES = ["teens", "twenties", "thirties", "fourties", "fifties"]
GENDERS = ["male_masculine", "female_feminine"]


def remove_duplicates_total_with_pandas(read_path: str, write_path: str):
    df = pd.read_csv(filepath_or_buffer=read_path, encoding="utf-8", header=0)
    df.drop_duplicates(subset=["client_id"])
    df.to_csv(header=df.columns, path_or_buf=write_path, mode="w")


def get_speakers(read_path: str, write_path: str):
    chunksize = 10000
    with pd.read_table(filepath_or_buffer=read_path, encoding="utf-8", sep="\\t", chunksize=chunksize, header=0) as reader:
        for chunk in reader:
            chunk = chunk[chunk['gender'].isin(GENDERS) & chunk['age'].isin(AGES)]
            chunk = chunk.drop_duplicates(subset=["client_id"])

            chunk.to_csv(header=chunk.columns, path_or_buf=write_path, mode="a", index=False)


def fix_encoding():
    with open(path_to_others, 'r', encoding='utf-8') as f:
        content = f.read()

    with open('../data/_others.txt', 'w', encoding="utf-8") as f:
        f.writelines(content)

fix_encoding()

path = PATH + VALIDATED

# get_speakers(path, "../data/Validated-Common-Voice-Speakers.csv")
remove_duplicates_total_with_pandas("../data/Validated-Common-Voice-Speakers.csv", "../data/Validated-Common-Voice-Speakers.csv")