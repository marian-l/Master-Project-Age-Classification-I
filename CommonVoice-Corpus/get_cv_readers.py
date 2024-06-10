import pandas as pd
import numpy as np

PATH = "D:/Sprachdaten/cv-corpus-17.0-2024-03-15-de/cv-corpus-17.0-2024-03-15/de/"
VALIDATED = "validated.tsv"
CLIP_DURATIONS = "clip_durations.tsv"
OTHERS = "other - Kopie.tsv"

path_to_validated = PATH + VALIDATED
path_to_others = PATH + OTHERS

speakers = []

def get_speakers():
    # with open(path_to_others, "r", encoding="windows-1252") as f:

    speakers = []
    temp_speakers = set()
    ages = []
    genders = []

    chunksize = 10000
    for chunk in pd.read_table(filepath_or_buffer="../data/_others.txt", encoding="utf-8", sep="\\t", chunksize=chunksize, header=0):
        for index, line in chunk.iterrows():
            speakers.append([line["client_id"], line["age"], line["gender"]])
            temp_speakers.add(line["client_id"])
            # temp_speakers.update([line["client_id"], line["age"], line["gender"]])
            # speakers.append(line["client_id"])
            # speakers.append({"client_id": line["client_id"], "age": line["age"], "gender": line["gender"]})

        speakers = np.unique(np.array([np.sort(sub) for sub in speakers]), axis=0)
        speakers = list(set(tuple(sub) for sub in speakers))
        temp_speakers = set(speakers[0])
        # speakers = set(speakers)
        print(speakers)

def fix_encoding():
    # Read the file with the incorrect encoding
    with open(path_to_others, 'r', encoding='iso-8859-1') as f:
        content = f.read()

    # Re-encode the content to UTF-8
    fixed_content = content.encode('latin_1').decode('utf-8')

    # Write the fixed content back to the file
    with open('../data/_others.txt', 'w', encoding="windows-1252") as f:
        f.writelines(fixed_content)


def fix_encoding2():
    with open(path_to_others, 'r', encoding='utf-8') as f:
        content = f.read()

    with open('../data/_others.txt', 'w', encoding="utf-8") as f:
        f.writelines(content)

fix_encoding2()
# fix_encoding()
get_speakers()