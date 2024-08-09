import pandas as pd
from matplotlib import pyplot as plt
import re

PATH = "D:/Sprachdaten/cv-corpus-17.0-2024-03-15-de/cv-corpus-17.0-2024-03-15/de/"
VALIDATED = "validated.tsv"
CLIP_DURATIONS = "clip_durations.tsv"
OTHERS = "other - Kopie.tsv"
TRAIN = "train.tsv"
TEST = "test.tsv"

path_to_validated = PATH + VALIDATED
path_to_others = PATH + OTHERS
path_to_train = PATH + TRAIN
path_to_test = PATH + TEST

AGES = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties"]
GENDERS = ["male_masculine", "female_feminine"]

_ages = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89"]


def get_gender_age_distribution_chart(read_path: str, write_path: str = ""):
    df = pd.read_csv(filepath_or_buffer=read_path, encoding="utf-8", header=0)

    gender_distribution = []
    for gender in GENDERS:
        temp_df = df[df["gender"] == gender]
        gender_distribution.append([gender, temp_df.shape[0]])

    age_distribution = []

    temp = 0
    for age in AGES:
        temp_df = df[df["age"] == age]
        age_distribution.append([_ages[temp], temp_df.shape[0]])
        temp += 1

    gender_count = 0
    age_count = 0

    for item in gender_distribution:
        gender_count = gender_count + item[1]
    for item in age_distribution:
        age_count = age_count + item[1]

    if age_count != gender_count:
        print("Data is malformed")

    age_distribution = dict(age_distribution)
    gender_distribution = dict(gender_distribution)

    plt.rcParams["figure.figsize"] = [11, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig, axs = plt.subplots(1, 2)
    axs[0].bar(list(age_distribution.keys()), list(age_distribution.values()))
    axs[1].bar(list(gender_distribution.keys()), list(gender_distribution.values()))

    plt.show()


def dataset_quality_filter(read_path: str, write_path: str = "", sep: str = None):
    chunksize = 10000
    if sep:
        df = pd.read_csv(filepath_or_buffer=read_path, encoding="utf-8", header=0, sep=sep)
    else:
        df = pd.read_csv(filepath_or_buffer=read_path, encoding="utf-8", header=0)

    # filter AGE: "age" and "nan"
    filtered = df["age"].isin(AGES)
    df = df[filtered]
    df.to_csv(header=df.columns, path_or_buf='../_datasets' + write_path + '/CV-FilteredAges.csv', mode="w")

    filtered = df['gender'].isin(GENDERS)
    df = df[filtered]
    df.to_csv(header=df.columns, path_or_buf='../_datasets' + write_path + '/CV-FilteredAgesAndGenders.csv', mode="w")

    # filter gender: "male_masculine" and "female_feminine"
    filtered = df['gender'].isin(['male_masculine'])
    male_df = df[filtered]
    male_df.to_csv(header=male_df.columns, path_or_buf='../_datasets' + write_path + '/male_CV-FilteredAgesAndGenders.csv', mode="a")

    filtered = df['gender'].isin(['female_feminine'])
    female_df = df[filtered]
    female_df.to_csv(header=female_df.columns, path_or_buf='../_datasets' + write_path + '/female_CV-FilteredAgesAndGenders.csv', mode="a")

    # free memory
    del df

    # filter duplicate entries to get total amount of unique speakers per gender
    male_df.drop_duplicates(keep='first', inplace=True, subset='client_id')
    male_df.to_csv(header=male_df.columns, path_or_buf='../_datasets' + write_path + '/male_CV-FilteredAges-UniqueSpeakers.csv', mode="a")

    del male_df

    female_df.drop_duplicates(keep='first', inplace=True, subset='client_id')
    female_df.to_csv(header=female_df.columns, path_or_buf='../_datasets' + write_path + '/female_CV-FilteredAges-UniqueSpeakers.csv',
                     mode="a")

    del female_df


def get_each_speakers_clips(read_path: str, write_path: str = "", sep: str = None):
    if sep != None:
        df = pd.read_csv(filepath_or_buffer=read_path, encoding='utf-8', sep=sep)
    else:
        df = pd.read_csv(filepath_or_buffer=read_path, encoding='utf-8', quotechar='"', sep=',')

    speakers_clips = {}

    for client_id, group in df.groupby('client_id'):
        clips = []
        for _, row in group.iterrows():
            clip_info = {
                'client_id': row['client_id'],
                'path': row['path'],
                'sentence_id': row['sentence_id'],
                'sentence': row['sentence'],
                'up_votes': row['up_votes'],
                'down_votes': row['down_votes'],
                'age': row['age'],
                'gender': row['gender'],
                'accents': row['accents'],
            }
            clips.append(clip_info)

        speakers_clips[client_id] = clips

    rows = []
    for speaker, clips in speakers_clips.items():
        for clip in clips:
            row = {'speaker_id': speaker}
            row.update(clip)
            rows.append(row)

    df = pd.DataFrame(rows)

    parts = re.split(r'[/\-]', read_path)
    if read_path.find("Remaining") != -1:
        current_info = f"{parts[3]}-{parts[4]}-{parts[5]}"
    else:
        current_info = f"{parts[3]}-{parts[4]}"
    # df.to_csv('../data/CV-AllSpeakers-CorrespondingClips.csv')
    df.to_csv('../_datasets' + write_path + current_info + '-CorrespondingClips.csv')


def get_train_datasets(read_path: str, write_path: str = ""):
    gender = ''

    # currently not sure how to best search for substring
    if read_path.find('female') != -1:
        gender = 'female'
    else:
        gender = 'male'

    df = pd.read_csv(filepath_or_buffer=read_path, encoding="utf-8", header=0)

    grouped = df.groupby('age')

    # need to take out 1, 5, 10, 20 and 50 speakers per age range
    drop_count = [1, 5, 10, 20, 50]

    # iterate over different amounts of rows to drop
    for drop in drop_count:
        current_train_set = []
        remainders_train_set = []

        # drop rows and append modified to current_train_set
        for age, group in grouped:
            deleted_entries = group
            try:
                modified_group = group.drop(group.index[0:drop])
                deleted_entries = group.drop(group.index[drop:group.size])

            except IndexError:
                modified_group = group.drop(group.index[0:group.size])

            current_train_set.append(modified_group)
            remainders_train_set.append(deleted_entries)

        current_train_set = pd.concat(current_train_set, ignore_index=True)
        current_train_set.to_csv(path_or_buf='../_datasets' + write_path + 'Trainsets/CV-' + gender + '-' + str(drop) + '-Speakers-Removed-Trainset.csv', header=True, index=True)

        remainders_train_set = pd.concat(remainders_train_set, ignore_index=True)
        remainders_train_set.to_csv(path_or_buf='../_datasets' + write_path + 'Trainsets/CV-' + gender + '-' + str(drop) + '-Speakers-Removed-Testset.csv', header=True, index=True)

    # iterate over different amounts of rows to drop
    for drop in drop_count:
        current_train_set = []
        remainders_train_set = []

        # drop rows and append modified to current_train_set
        for age, group in grouped:
            deleted_entries = group
            try:
                modified_group = group.drop(group.index[drop:group.size])
                deleted_entries = group.drop(group.index[0:drop-1])

            except IndexError:
                modified_group = group.drop(group.index[0:group.size])

            current_train_set.append(modified_group)
            remainders_train_set.append(deleted_entries)

        current_train_set = pd.concat(current_train_set, ignore_index=True)
        current_train_set.to_csv(path_or_buf='../_datasets' + write_path + 'Trainsets/CV-' + gender + '-' + str(drop) + '-Speakers-Remaining-Trainset.csv', header=True, index=True)

        remainders_train_set = pd.concat(remainders_train_set, ignore_index=True)
        remainders_train_set.to_csv(path_or_buf='../_datasets' + write_path + 'Trainsets/CV-' + gender + '-' + str(drop) + '-Speakers-Remaining-Testset.csv', header=True, index=True)


# TODO alle zu einem datensatz gelöschten Speaker in einem separaten Dataset
# TODO alle zu einem datensatz gehörigen Audioclips + Alterskategorien

def find_intersecting_speakers(set1: str, set2: str):
    df1 = pd.read_csv(set1)
    df2 = pd.read_csv(set2)

    validated_speakers = set(df1['client_id'].unique())
    test_speakers = set(df2['client_id'].unique())

    overlapping_speakers = validated_speakers.intersection(test_speakers)

    if overlapping_speakers:
        print("Overlapping speakers found between " + set1 + " and " + set2)
    else:
        print("No overlapping speakers found between " + set1 + " and " + set2)


def finished_tasks():
    pass
    # get_union_dfs_val_train()

    # get_gender_age_distribution_chart("../_datasets/CommonVoice/_Union_Train-Validated/unique_male_speakers.csv")
    # get_gender_age_distribution_chart("../_datasets/CommonVoice/_Union_Train-Validated/unique_female_speakers.csv")

    # do_all("D:\\Sprachdaten\\cv-corpus-17.0-2024-03-15-de\\cv-corpus-17.0-2024-03-15\\de\\train.tsv")

    # fix_encoding()
    # get_speakers(path, "../data/Validated-Common-Voice-Speakers.csv")
    # remove_duplicates_total_with_pandas("../data/Validated-Common-Voice-Speakers.csv", "../data/Validated-Common-Voice-Speakers.csv")

    # get all clips from all speakers
    # get_each_speakers_clips(read_path=PATH + VALIDATED)

    # filter the CV-Corpus for quality
    # dataset_quality_filter(read_path='../data/CV-AllSpeakers-CorrespondingClips.csv')

    ###### get the sub-datasets
    # dataset_quality_filter(read_path=PATH + 'validated.tsv', sep='\t', write_path='/CommonVoice/Validated.tsv')
    # dataset_quality_filter(read_path=PATH + 'train.tsv', sep='\t', write_path='/CommonVoice/Train.tsv')
    # dataset_quality_filter(read_path=PATH + 'test.tsv', sep='\t', write_path='/CommonVoice/Test.tsv')
    # dataset_quality_filter(read_path=PATH + 'dev.tsv', sep='\t', write_path='/CommonVoice/Dev.tsv')

    ##### get the gender and age distribution charts for male and female
    # get_gender_age_distribution_chart("../data/Validated-Common-Voice-Speakers.csv")
    # get_gender_age_distribution_chart("../data/Validated.tsv/male_CV-FilteredAges-UniqueSpeakers.csv")
    # get_gender_age_distribution_chart("../data/Validated.tsv/female_CV-FilteredAges-UniqueSpeakers.csv")

    ##### build a number of datasets with different filter-sizes
    # get_train_datasets("../_datasets/CommonVoice/Train.tsv/female_CV-FilteredAges-UniqueSpeakers.csv", '/CommonVoice/Train.tsv/')
    # get_train_datasets("../_datasets/CommonVoice/Train.tsv/male_CV-FilteredAges-UniqueSpeakers.csv", '/CommonVoice/Train.tsv/')

    # get_train_datasets("../_datasets/CommonVoice/Validated.tsv/female_CV-FilteredAges-UniqueSpeakers.csv", '/CommonVoice/Validated.tsv/')
    # get_train_datasets("../_datasets/CommonVoice/Validated.tsv/male_CV-FilteredAges-UniqueSpeakers.csv", '/CommonVoice/Validated.tsv/')

    ##### check for unique-ness of speakers in between different source files:
    ## TRAIN.TSV || VALIDATED.TSV
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/Validated.tsv/female_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Train.tsv/female_CV-FilteredAges-UniqueSpeakers.csv')
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/Validated.tsv/male_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Train.tsv/male_CV-FilteredAges-UniqueSpeakers.csv')

    # ## TRAIN.TSV || TEST.TSV
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/Test.tsv/female_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Train.tsv/female_CV-FilteredAges-UniqueSpeakers.csv')
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/Test.tsv/male_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Train.tsv/male_CV-FilteredAges-UniqueSpeakers.csv')

    # ## VALIDATED.TSV || TEST.TSV
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/Validated.tsv/female_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Test.tsv: str/female_CV: str-FilteredAges-UniqueSpeakers.csv')
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/Validated.tsv/male_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Test.tsv/male_CV-FilteredAgesUniqueSpeakers.csv')

    ## TRAIN.TSV || dev.tsv
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/dev.tsv/female_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Train.tsv/female_CV-FilteredAges-UniqueSpeakers.csv')
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/dev.tsv/male_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Train.tsv/male_CV-FilteredAges-UniqueSpeakers.csv')

    # ## TRAIN.TSV || TEST.TSV
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/Test.tsv/female_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Train.tsv/female_CV-FilteredAges-UniqueSpeakers.csv')
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/Test.tsv/male_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Train.tsv/male_CV-FilteredAges-UniqueSpeakers.csv')

    # ## dev.tsv || TEST.TSV
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/dev.tsv/female_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Test.tsv/female_CV-FilteredAges-UniqueSpeakers.csv')
    # find_intersecting_speakers(set1='../_datasets/CommonVoice/dev.tsv/male_CV-FilteredAges-UniqueSpeakers.csv',
    #                            set2='../_datasets/CommonVoice/Test.tsv/male_CV-FilteredAges-UniqueSpeakers.csv')
    ##### overlap between:
        # VALIDATED.TSV + TEST.TSV; VALIDATED.TSV + TRAIN.TSV

#########################

    ##### Merge dev.tsv and train.tsv
    # df_1 = pd.read_csv("../_datasets/CommonVoice/Dev.tsv/male_CV-FilteredAges-UniqueSpeakers.csv")
    # df_2 = pd.read_csv("../_datasets/CommonVoice/Train.tsv/male_CV-FilteredAges-UniqueSpeakers.csv")

    # unique_male_speakers_union_df = pd.concat([df_1, df_2]).drop_duplicates().reset_index(drop=True)

    # df_1 = pd.read_csv("../_datasets/CommonVoice/Dev.tsv/female_CV-FilteredAges-UniqueSpeakers.csv")
    # df_2 = pd.read_csv("../_datasets/CommonVoice/Train.tsv/female_CV-FilteredAges-UniqueSpeakers.csv")

    # unique_female_speakers_union_df = pd.concat([df_1, df_2]).drop_duplicates().reset_index(drop=True)

    # unique_female_speakers_union_df.to_csv(
    #     path_or_buf="../_datasets/CommonVoice/dev-train-set/unique-female-speakers.csv",
    #     encoding="UTF-8", header=True)
    # unique_male_speakers_union_df.to_csv(path_or_buf="../_datasets/CommonVoice/dev-train-set/unique-male-speakers.csv",
    #                                      encoding="utf-8", header=True)

    # df_1 = pd.read_csv("../_datasets/CommonVoice/Dev.tsv/male_CV-FilteredAgesAndGenders.csv")
    # df_2 = pd.read_csv("../_datasets/CommonVoice/Train.tsv/male_CV-FilteredAgesAndGenders.csv")

    # unique_male_speakers_union_df = pd.concat([df_1, df_2]).drop_duplicates().reset_index(drop=True)

    # df_1 = pd.read_csv("../_datasets/CommonVoice/Dev.tsv/female_CV-FilteredAgesAndGenders.csv")
    # df_2 = pd.read_csv("../_datasets/CommonVoice/Train.tsv/female_CV-FilteredAgesAndGenders.csv")

    # unique_female_speakers_union_df = pd.concat([df_1, df_2]).drop_duplicates().reset_index(drop=True)

    # unique_female_speakers_union_df.to_csv(
    #     path_or_buf="../_datasets/CommonVoice/dev-train-set/female-speakers-clips.csv",
    #     encoding="UTF-8", header=True)
    # unique_male_speakers_union_df.to_csv(path_or_buf="../_datasets/CommonVoice/dev-train-set/male-speakers-clips.csv",
    #                                      encoding="utf-8", header=True)

    # get_gender_age_distribution_chart('../_datasets/CommonVoice/dev-train-set/unique-male-speakers.csv')
    # get_gender_age_distribution_chart('../_datasets/CommonVoice/dev-train-set/unique-female-speakers.csv')

    # trainsets = glob.glob("../data/Trainsets/*.csv")
    # for file in trainsets:
    #     get_each_speakers_clips(file)

    # df = pd.read_csv(filepath_or_buffer='../data/Trainsets/CV-female-50-Trainset.csv')
    # df = pd.read_csv(filepath_or_buffer='../data/Trainsets/CV-male-50-Trainset.csv')
    # df = pd.read_csv(filepath_or_buffer='../data/Trainsets/CV-male-50-Trainset.csv')

    import glob

# get_each_speakers_clips(read_path='../_datasets/CommonVoice/dev-train-set/male-speakers-clips.csv', write_path='/CommonVoice/dev-train-set/')
# get_each_speakers_clips(read_path='../_datasets/CommonVoice/dev-train-set/female-speakers-clips.csv', write_path='/CommonVoice/dev-train-set/')


