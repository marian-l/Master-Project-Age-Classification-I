import glob
import os
import pandas as pd
import xml.etree.ElementTree as ET


def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = {
        "client_id": root.findtext("speaker_id"),
        "gender": root.findtext("gender"),
        "age": root.findtext("ageclass"),
        "sentence_id": root.findtext("sentence_id"),
    }

    return data


def parse_multiple_xml(files):
    records = []

    for file in files:
        record = parse_xml(file)
        records.append(record)

    return pd.DataFrame(records)


def get_distribution_of_speakers():
    global directory_path
    xml_files = glob.glob(directory_path + '\*.xml')

    # Parse the XML files and create a DataFrame
    df = parse_multiple_xml(xml_files)

    # Display the DataFrame
    print(df.head())

    df.drop_duplicates(subset='speaker_id', inplace=True)

    print("")

# directory_path = "D:\Sprachdaten\german-speechdata-package-v2\german-speechdata-package-v2\\train"
# xml_files = glob.glob(directory_path + '\*.xml')
# df = parse_multiple_xml(xml_files)
# df.to_csv(path_or_buf='../_datasets/GermanSpeechDat/Train/train.csv')


# Directory containing the XML files



# df.to_csv(path_or_buf='../data/GermanSpeechDat/Train/AllMetaData.csv')
