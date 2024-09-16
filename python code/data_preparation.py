import pandas
from datasets import Dataset, Audio, Features, ClassLabel, Value

ACCENTS = ['nan', 'Kanadisches Deutsch', 'Deutschland Deutsch,leicht Berlinerisch', 'Hochdeutsch', 'Deutschland Deutsch,Alemanischer Akzent,Süddeutscher Akzent', 'Finnisch Deutsch', 'Standarddeutsch,Ruhrpott,Deutschland Deutsch', 'Deutschland Deutsch,Fränkisch', 'Deutschland Deutsch,Türkisch Deutsch', 'Brasilianisches Deutsch', 'Französisch Deutsch', 'Deutschland Deutsch,Hochdeutsch', 'Italienisch Deutsch', 'Schweizerdeutsch,Zürichdeutsch', 'Niedersächsisches Deutsch,Deutschland Deutsch', 'liechtensteinisches Deutscher', 'Österreichisches Deutsch,Kärnten,Steiermark', 'Österreichisches Deutsch,Oberösterreichisches Deutsch', 'Deutschland Deutsch,Schwäbisch', 'Slowakisch Deutsch', 'Deutschland Deutsch,Nordrhein-Westfalen', 'Deutschland Deutsch', 'Deutschland Deutsch,Nordrhein Westfalen', 'Österreichisches Deutsch,Theaterdeutsch,Wienerisch,Burgenländisch (Süden),Niederösterreich (Mödling)', 'Akzentfrei', 'Österreichisches Deutsch,Bayern', 'Deutschland Deutsch,Ruhrgebiet Deutsch,West Deutsch', 'Polnisch Deutsch', 'Belgisches Deutsch', 'Französisch Deutsch,Deutschland Deutsch', 'Niederländisch Deutsch', 'Deutschland Deutsch,Österreichisches Deutsch', 'Österreichisches Deutsch', 'Hochdeutsch,Deutschland Deutsch', 'Deutschland Deutsch,Brandenburger Dialekt,Berliner Dialekt', 'Deutschland Deutsch,Niederrhein', 'starker lettischer Akzent', 'Badisch,Allemannisch', 'Russisch Deutsch', 'Deutschland Deutsch,Französisch Deutsch', 'Deutsch/Berlinern,Berlinerisch,klar,zart,feminin', 'Schweizerdeutsch,Deutschland Deutsch', 'Ruhrpott Deutsch', 'Österreichisches Deutsch,Lower Austria', 'Ungarisch Deutsch', 'Amerikanisches Deutsch', 'Slowenisch Deutsch', 'Schwäbisch Deutsch,Deutschland Deutsch', 'Deutschland Deutsch,Berliner Deutsch', 'Deutschland Deutsch,Standard ', 'Alemannische Färbung,Schweizer Standart Deutsch', 'Leichter saarländische Einschlag mit Unschärfe bei ch und sch,Deutschland Deutsch', 'Israeli', 'Türkisch Deutsch', 'Deutschland Deutsch,Saarland Deutsch,Plattdeutsch,Saarländisch', 'Deutschland Deutsch,Norddeutsch', 'Schweizerdeutsch', 'Süddeutsch', 'Deutschland Deutsch,Akzentfrei', 'Luxemburgisches Deutsch', 'Griechisch Deutsch', 'Belgisches Deutsch,Französisch Deutsch', 'Tschechisch Deutsch', 'Deutschland Deutsch,Süddeutsch', 'Nordrhein-Westfalen,Bundesdeutsch, Hochdeutsch,Deutschland Deutsch', 'Britisches Deutsch', 'Deutschland Deutsch,Britisches Deutsch', 'Deutschland Deutsch,sächsisch']

audio_df = pandas.read_csv(filepath_or_buffer='../_datasets/CommonVoice/dev-train-set/all-speakers-clips.csv', header=0, encoding='utf-8', index_col=None)

# audio_df = pandas.read_csv(filepath_or_buffer='../_datasets/CommonVoice/dev-train-set/male-speakers-clips.csv',header=0, encoding='utf-8')
audio_df = audio_df[['path', 'age', 'gender', 'client_id', 'accents']]

audio_df['path'] = 'D:\Sprachdaten\cv-corpus-17.0-2024-03-15-de\cv-corpus-17.0-2024-03-15\de\clips\\' + audio_df['path']

features = Features({
    'path': Audio(sampling_rate=16000),
    'age':
        # ClassLabel(num_classes=8),
        ClassLabel(names=["teens", "twenties", "thirties", "fourties", "fifties", "sixties", "seventies", "eighties"]),
        # Value('int8'),
    'gender': ClassLabel(names=['male_masculine', 'female_feminine']),
    'client_id': Value(dtype='string'),
    'accents': ClassLabel(names=ACCENTS),
})

def sample_clips_from_full(df, SAMPLE_SIZE):
    # result_df = df.groupby('client_id').filter(lambda x: len(x) >= 2) # unklar, was hier genau gefiltert wird

    result_df = df.groupby('client_id').apply(lambda x: x.sample(SAMPLE_SIZE, replace=True)).reset_index(drop=True)
    return result_df

def sampled_by_client_id():
    for i in range (1,10):
        print(i)
        if i == 5:
            continue

        SAMPLE_SIZE = i # how many clips of each speaker are sampled

        sampled_set = audio_df
        sampled_set = sample_clips_from_full(sampled_set, SAMPLE_SIZE)

        # dataset = Dataset.from_pandas(audio_df, features=features)
        dataset = Dataset.from_pandas(sampled_set, features=features)

        dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

        print(dataset)

        dataset.save_to_disk(dataset_path="../_datasets/CommonVoice/dev-train-set/sampled_dataset_SIZE_"+str(i)+".arrow")

def sampled_by_age_class():
    sampled_set = audio_df

    # df = sampled_set.groupby('age').apply(lambda x: x.sample(500, replace=True)).reset_index(drop=True)
    #group_lengths = sampled_set.groupby('age').size()
    #print(group_lengths)
    # sampled_set.reset_index(drop=True)

    result_df = sampled_set.groupby('age').apply(lambda x: x.sample(500, replace=True)).reset_index(drop=True)
    result_df.drop_duplicates(inplace=True, subset='path')

    group_lengths = result_df.groupby('age').size()
    print(group_lengths)

    # dataset = Dataset.from_pandas(result_df, features=features, preserve_index=False)

    # dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # dataset.save_to_disk(dataset_path="../_datasets/CommonVoice/dev-train-set/500x-per-age-class.arrow")

sampled_by_age_class()
# sampled_by_client_id()
