import sys
import io
import glob
def filter_german_speakers():
    with open("VoxTube-Speakers.txt", "r") as f:
        lines = f.readlines()

        # Filter lines that contain the word 'german'
        filtered_lines = [line for line in lines if "german" in line.lower()]

        # Write the filtered lines back to the file
        with open("VoxTube-Speakers-German.txt", "x") as f:
            f.writelines(filtered_lines)

def filter_speakers_id():
    with open("VoxTube-Speakers-German.txt", "r") as f:
        lines = f.readlines()

    filtered_lines = [line.replace(",female,german", "") if "female" in line else line.replace(",male,german", "")
                          for line in lines]

    f.close()

    with open("VoxTube-Speakers-German.txt", "w") as f:
        f.writelines(filtered_lines)

filter_german_speakers()
filter_speakers_id()