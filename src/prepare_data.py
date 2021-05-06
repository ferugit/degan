import os
import datetime
import argparse

import json
import pandas as pd
from collections import Counter


def create_dataframe(args, columns):

    # Start scanning data and saving information in a dataframe.
    dataframe = pd.DataFrame(columns=columns)

    # Create sample ID counter
    sample_id = 0

    print("Start walking databases...")

    # Scan UrbanSound8K database
    if(args.use_urbansound):

        print("\ti) Reading urbansound dataset...")

        # Read urbansound tsv: only wanted columns
        urbansound_path = os.path.join(args.src, 'urbansound.tsv')
        assert os.path.isfile(str(urbansound_path)), " urbansound.tsv file does not exist - '{d}'".format(d=urbansound_path)
        urbansound_df = pd.read_csv(urbansound_path, sep='\t', header=0, usecols=columns)

        # Get number of samples
        n_samples = len(urbansound_df.index)

        # Replace database 'Sample_ID' with and incremented ID
        urbansound_df['Sample_ID'] = pd.Series(range(sample_id, sample_id + n_samples))

        # Append to dataframe list
        dataframe = dataframe.append(urbansound_df)

        # Update samples_id
        sample_id += n_samples

    if(args.use_esc50):

        print("\tii) Reading ESC-50 dataset...")

        # Read urbansound tsv: only wanted columns
        esc50_path = os.path.join(args.src, 'esc50.tsv')
        assert os.path.isfile(str(esc50_path)), " esc50.tsv file does not exist - '{d}'".format(d=esc50_path)
        esc50_df = pd.read_csv(esc50_path, sep='\t', header=0, usecols=columns)

        # Get number of samples
        n_samples = len(esc50_df.index)

        # Replace database 'Sample_ID' with and incremented ID
        esc50_df['Sample_ID'] = pd.Series(range(sample_id, sample_id + n_samples))

        # Append to dataframe list
        dataframe = dataframe.append(esc50_df)

        # Update samples_id
        sample_id += n_samples


    return dataframe


def get_class_distribution(dataframe, class_name, normalize=True):
    return dataframe[class_name].value_counts(normalize=normalize, dropna=False).to_dict()


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

    # Sound Type Information information
    samples_info_train = train_df['Class'].value_counts().to_dict()
    samples_info_dev = dev_df['Class'].value_counts().to_dict()
    samples_info_total = Counter(samples_info_train)
    samples_info_total.update(samples_info_dev)
    samples_info_total = dict(samples_info_total)

    information = {
        'General': {
            'Number of samples' : sample_number_total,
            'Hours of audio' : audio_hours_wuw_total,
            'Samples information': samples_info_total
        },
        'Train': {
            'Number of samples' : sample_number_train,
            'Hours of audio' : audio_hours_wuw_train,
            'Sample percentage' : sample_percentage_train,
            'Samples information' : samples_info_train
        },
        'Validation': {
            'Number of samples' : sample_number_dev,
            'Hours of audio' : audio_hours_wuw_dev,
            'Sample percentage' : sample_percentage_dev,
            'Samples information' : samples_info_dev
        }
    }

    return information


def generate_kfold_partition(dataframe, fold):
    ''' 
    The criteria for generating the k-fold partition is the following:
        - Use given k-fold split to generate partitions
    '''
    dev_df = dataframe[dataframe['Fold'] == fold]
    dev_paths = dev_df['Sample_Path']
    train_df = dataframe[~dataframe['Sample_Path'].isin(dev_paths)]

    return train_df, dev_df

def get_classes_index(dataframe):
    column_values = dataframe[["Class", "Label"]]. values. ravel()
    unique_values = pd.unique(column_values)
    classes_index = {}

    for i in range(len(unique_values)):
        if i % 2 == 0:
            classes_index[unique_values[i+1]] = unique_values[i]

    return classes_index


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
    columns = ['Sample_ID', 'Sample_Path', 'Audio_Length', 'Fold', 'Class', 'Label', 'Database']
    print('Global columns that will be used: ' + str(columns))

    # Get global dataframe
    dataframe = create_dataframe(args, columns)

    # Store global dataframe
    dataframe.to_csv(os.path.join(partition_dirpath, "dataset.tsv"), sep="\t", index=None)

    # Get and save classes index
    classes_index = get_classes_index(dataframe)
    with open(partition_dirpath + '/classes_index.json', 'w', encoding='utf-8') as f:
        json.dump(classes_index, f, ensure_ascii=False, indent=4)

    # Generate partitions
    train_df, dev_df = generate_kfold_partition(dataframe, args.fold)

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

    # Datasets to use
    parser.add_argument('--use_urbansound', dest='use_urbansound', action='store_true', help='use UrbanSound8K dataset')
    parser.add_argument('--use_esc50', dest='use_esc50', action='store_true', help='use ESC-50 dataset')

    # Fold to test
    parser.add_argument('--fold', type=int, default=10, metavar='F', help='fold to use for testing')

    args = parser.parse_args()

    # Run main
    main(args)