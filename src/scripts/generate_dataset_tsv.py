import sys
sys.path.insert(1, sys.path[0].replace('/src/scripts', ''))

import os
import argparse
import pandas as pd
import torchaudio


def get_length(sample_path):
    audio, sr = torchaudio.load(sample_path)
    length = audio.shape[1]/sr
    return length


def main(args):
    """
    This function walks over the data paths to create a TSV that constains all the audio metadata.
    """

    torchaudio.set_audio_backend("sox_io")

    # Data path
    wavs_path = os.path.join(args.src, 'impulse_responses_2')

    # Create dataframe list
    dataframe_list = []

    for audio_file in os.listdir(wavs_path):
        if(audio_file.endswith(".wav") and not audio_file.startswith('.')):

            # Sample ID
            sample_id = audio_file

            # Sample path
            sample_path = os.path.join(wavs_path, audio_file)

            # Audio Length
            try:
                audio_length = get_length(sample_path) 
            except:
                print('Non valid file:' + sample_path)
                continue

            # TODO: get impulse response type

            # Write row on dataframe
            dataframe_list.append([ sample_id, sample_path, audio_length, 'im'])


    # Build audioset tsv file
    audioset_df = pd.DataFrame(dataframe_list, columns=['Sample_ID', 'Sample_Path', 'Audio_Length', 'Database'])
    audioset_df.to_csv('data/tsv/impulse_responses.tsv', sep = '\t', index=None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scrip to create Impulse Responses tsv")

    # Source Impulse Responses data placed in the data folder of the project 
    parser.add_argument("--src", help="source directory", default="data/")
    args = parser.parse_args()

    # Run main
    main(args)