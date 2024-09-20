import pandas as pd

def get_union_dfs_val_train():
    df_1 = pd.read_csv("../_datasets/CommonVoice/Validated.tsv/male_CV-FilteredAges-UniqueSpeakers.csv")
    df_2 = pd.read_csv("../_datasets/CommonVoice/Train.tsv/male_CV-FilteredAges-UniqueSpeakers.csv")

    unique_male_speakers_union_df = pd.concat([df_1, df_2]).drop_duplicates().reset_index(drop=True)

    df_1 = pd.read_csv("../_datasets/CommonVoice/Validated.tsv/female_CV-FilteredAges-UniqueSpeakers.csv")
    df_2 = pd.read_csv("../_datasets/CommonVoice/Train.tsv/female_CV-FilteredAges-UniqueSpeakers.csv")

    unique_female_speakers_union_df = pd.concat([df_1, df_2]).drop_duplicates().reset_index(drop=True)

    unique_female_speakers_union_df.to_csv(path_or_buf="../_datasets/CommonVoice/_Union_Train-Validated/unique_female_speakers.csv",
                                           encoding="UTF-8", header=True)
    unique_male_speakers_union_df.to_csv(path_or_buf="../_datasets/CommonVoice/_Union_Train-Validated/unique_male_speakers.csv",
                                         encoding="UTF-8", header=True)