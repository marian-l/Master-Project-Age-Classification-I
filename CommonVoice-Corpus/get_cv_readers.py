import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
    with pd.read_table(filepath_or_buffer=read_path, encoding="utf-8", sep="\\t", chunksize=chunksize,
                       header=0) as reader:
        for chunk in reader:
            chunk = chunk[chunk['gender'].isin(GENDERS) & chunk['age'].isin(AGES)]
            chunk = chunk.drop_duplicates(subset=["client_id"])

            chunk.to_csv(header=chunk.columns, path_or_buf=write_path, mode="a", index=False)


def fix_encoding():
    with open(path_to_others, 'r', encoding='utf-8') as f:
        content = f.read()

    with open('../data/_others.txt', 'w', encoding="utf-8") as f:
        f.writelines(content)


def get_audios_from_speaker(read_path: str, write_path: str):
    chunksize = 10000
    with pd.read_table(filepath_or_buffer=read_path, encoding="utf-8", sep="\\t", chunksize=chunksize,
                       header=0) as reader:
        for chunk in reader:
            chunk = chunk[chunk['gender'].isin(GENDERS) & chunk['age'].isin(AGES)]

            chunk.to_csv(header=chunk.columns, path_or_buf=write_path, mode="a", index=False)


def get_gender_age_distribution_chart(read_path: str):
    df = pd.read_csv(filepath_or_buffer=read_path, encoding="utf-8", header=0)

    gender_distribution = []
    for gender in GENDERS:
        temp_df = df[df["gender"] == gender]
        gender_distribution.append([gender, temp_df.size])

    age_distribution = []
    for age in AGES:
        temp_df = df[df["age"] == age]
        age_distribution.append([age, temp_df.size])

    gc = 0
    ac = 0

    for item in gender_distribution:
        gc = gc + item[1]
    for item in age_distribution:
        ac = ac + item[1]

    if ac != gc:
        print("Data is malformed")

    age_distribution = dict(age_distribution)
    gender_distribution = dict(gender_distribution)

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig, axs = plt.subplots(1, 2)
    axs[0].bar(list(age_distribution.keys()), list(age_distribution.values()))
    axs[1].bar(list(gender_distribution.keys()), list(gender_distribution.values()))

    plt.show()


path = PATH + VALIDATED

# fix_encoding()
# get_speakers(path, "../data/Validated-Common-Voice-Speakers.csv")
# remove_duplicates_total_with_pandas("../data/Validated-Common-Voice-Speakers.csv", "../data/Validated-Common-Voice-Speakers.csv")
# get_audios_from_speaker(path, "../data/Validated-Common-Voice-ClipsPerSpeaker.csv")

get_gender_age_distribution_chart("../data/Validated-Common-Voice-Speakers.csv")
print("")
