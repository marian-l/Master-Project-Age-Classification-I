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

def get_speakers():
    # with open(path_to_others, "r", encoding="windows-1252") as f:

    speakers = []
    temp_speakers = []

    chunksize = 10000
    # with pd.read_table(filepath_or_buffer=path_to_validated, encoding="utf-8", sep="\\t", chunksize=chunksize, header=0) as reader:
    with pd.read_table(filepath_or_buffer="../data/_others.txt", encoding="utf-8", sep="\\t", chunksize=chunksize, header=0) as reader:
        for chunk in reader:
            chunk = chunk[chunk['gender'].isin(GENDERS) & chunk['age'].isin(AGES)]

            temp_speakers.append([chunk["client_id"], chunk["age"], chunk["gender"]])

        # temp_speakers = np.squeeze(np.unique(np.array([np.sort(sub) for sub in temp_speakers]), axis=0))
        # speakers.append(temp_speakers)
        # speakers = pd.concat(chunk)

    speakers = speakers.drop_duplicates().sort_values(by=['client_id', 'age', 'gender'])
    #speakers = np.squeeze(np.unique(np.array([np.sort(sub) for sub in speakers]), axis=0))

    # speakers = pd.DataFrame(data=speakers, columns=["client_id", "age", "gender"])
    speakers.to_csv(header=speakers.columns, path_or_buf="../data/Others2-CommonVoice-Speakers.csv", mode="a")

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