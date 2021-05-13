import os
import datetime
import argparse

import json
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split

def create_dataframe(args, columns):

    # Start scanning data and saving information in a dataframe.
    dataframe = pd.DataFrame(columns=columns)

    # Create sample ID counter
    sample_id = 0

    print("Start walking data...")

    print("\ti) Reading impulse responses...")

    # Read impulse responses tsv: only wanted columns
    ir_path = os.path.join(args.src, 'impulse_responses.tsv')
    assert os.path.isfile(str(ir_path)), " impulse_responses.tsv file does not exist - '{d}'".format(d=ir_path)
    impulse_responses_df = pd.read_csv(ir_path, sep='\t', header=0, usecols=columns)

    # Get number of samples
    n_samples = len(impulse_responses_df.index)

    # Replace database 'Sample_ID' with and incremented ID
    impulse_responses_df['Sample_ID'] = pd.Series(range(sample_id, sample_id + n_samples))

    # Append to dataframe list
    dataframe = dataframe.append(impulse_responses_df)

    # Update samples_id
    sample_id += n_samples

    return dataframe


def generate_partition_information(train_df, dev_df):

    # Get metrics about the number of samples
    sample_number_train = len(train_df.index)
    sample_number_dev = len(dev_df.index)
    sample_number_total = sample_number_train + sample_number_dev

    sample_percentage_train = (sample_number_train/sample_number_total)*100
    sample_percentage_dev = (sample_number_dev/sample_number_total)*100

    # Get hours information
    audio_hours_wuw_train = (train_df["Audio_Length"].sum())/3600.0
    audio_hours_wuw_dev = (dev_df["Audio_Length"].sum())/3600.0
    audio_hours_wuw_total = audio_hours_wuw_train + audio_hours_wuw_dev

    information = {
        'General': {
            'Number of samples' : sample_number_total,
            'Hours of audio' : audio_hours_wuw_total
        },
        'Train': {
            'Number of samples' : sample_number_train,
            'Hours of audio' : audio_hours_wuw_train,
            'Sample percentage' : sample_percentage_train
        },
        'Validation': {
            'Number of samples' : sample_number_dev,
            'Hours of audio' : audio_hours_wuw_dev,
            'Sample percentage' : sample_percentage_dev
        }
    }

    return information


def get_random_partition(dataframe):
    train_df, dev_df = train_test_split(dataframe, test_size=0.1)
    return train_df, dev_df


def main(args):

    # Get partition name
    if(args.name == ""):
        date = str(datetime.datetime.now()).replace(" ", "").replace("-", "").replace(":", "").split(".")[0]
        partition_dirpath = os.path.join(args.dst, "partition_" + date)
    else:
        partition_dirpath = os.path.join(args.dst, "partition_" + args.name.lower().replace(" ", "_"))

    print("New partition will be generated in: " + partition_dirpath)

    # Create destination directory
    if not os.path.exists(partition_dirpath):
        os.makedirs(partition_dirpath)

    # Define global columns for the partitions
    columns = ['Sample_ID', 'Sample_Path', 'Audio_Length', 'Database']
    print('Global columns that will be used: ' + str(columns))

    # Get global dataframe
    dataframe = create_dataframe(args, columns)

    # Store global dataframe
    dataframe.to_csv(os.path.join(partition_dirpath, "dataset.tsv"), sep="\t", index=None)

    # Generate partitions:
    train_df, dev_df = get_random_partition(dataframe)

    # Generate partition information
    partition_info = generate_partition_information(train_df, dev_df)

    # Save dataframes to tsv
    train_df.to_csv(os.path.join(partition_dirpath, "train.tsv"), sep='\t', index=None)
    dev_df.to_csv(os.path.join(partition_dirpath, "dev.tsv"), sep='\t', index=None)

    # Save partitions information
    with open(partition_dirpath + '/partition.json', 'w', encoding='utf-8') as f:
        json.dump(partition_info, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    # Read arguments
    parser = argparse.ArgumentParser(description="Audio Classification partitions generation")

    # Partitions parameters
    parser.add_argument("--src", help="source directory, default 'data/tsv'", default="data/tsv")
    parser.add_argument("--dst", help="destination directory, default 'partitions'", default="partitions")
    parser.add_argument("--name", help="partition folder name, if not set it will be named with a timestamp", default="")

    args = parser.parse_args()

    # Run main
    main(args)