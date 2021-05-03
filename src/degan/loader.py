import math
import random

import torch
import numpy as np
import pandas as pd

import utils.audioutils as au


def load_train_partitions(path, window_size=24000, fs=16000, augments=None):
    
    # Load train set
    train_df = pd.read_csv(path + 'train.tsv', sep='\t')
    train_ID_list = list(train_df['Sample_ID'])

    # Load validation set
    validation_df = pd.read_csv(path + 'dev.tsv', sep='\t')
    validation_ID_list = list(validation_df['Sample_ID'])

    # Generate Datasets
    train_dataset = AudioDataset(
        train_ID_list,
        train_df,
        window_size=window_size,
        fs=fs,
        augments=augments
        )
    validation_dataset = AudioDataset(
        validation_ID_list,
        validation_df,
        window_size=window_size,
        fs=fs,
        augments=augments
        )

    return train_dataset, validation_dataset


class AudioDataset(torch.utils.data.Dataset):
    """
    Torch dataset for lazy load.
    """
    def __init__(self, list_IDs, dataframe, window_size=24000, fs=16000, augments=None):
        
        self.window_size = window_size
        self.fs = fs # Hz

        # Data information
        self.list_IDs = list_IDs
        self.dataframe = dataframe
        self.n_samples = len(list_IDs)

        # Data augments
        self.augments = [k for k,v in augments.items() if v == True]
        self.white_noise = True if augments['white_noise'] else None

    def __len__(self):
        """
        Denote dataset sample.
        """
        return len(self.list_IDs)

    def get_number_of_classes(self):
        return self.dataframe['Class'].nunique()

    def __repr__(self):
        """
        Data infromation
        """
        repr_str = (
            "Number of samples: " + str(self.n_samples) + "\n"
            "Window size: " + str(self.window_size) + "\n"
            "Augments: " + str(self.augments) + "\n"
            "Databases: " + str(np.unique(self.dataframe['Database'])) + "\n"

        )
        return repr_str
        
    def __getitem__(self, index):
        """
        Get a single sample
        Args:
            index: index to recover a single sample
        Returns:
            x,y: features extracted and label
        """
        # Select sample
        ID = self.list_IDs[index]

        # Read audio
        audio_path = self.dataframe.set_index('Sample_ID').at[ID, 'Sample_Path']
        audio = self.__read_wav(audio_path)
        
        # Prepare audio
        audio = self.__prepare_audio(audio)

        label = self.dataframe.set_index('Sample_ID').at[ID, 'Label']

        return ID, audio, label

    def __read_wav(self, filepath):
        """
        Read audio wave file applying normalization with respecto of the maximum of the signal
        Args:
            filepath: audio file path
        Returns:
            audio_signal: numpy array containing audio signal
        """
        _, audio_signal = au.open_wavfile(filepath)
        audio_signal = self.__normalize_audio(audio_signal)
        return audio_signal
    
    def __normalize_audio(self, audio, eps=0.001):
        """
        Peak normalization.
        """
        return (audio.astype(np.float32) / float(np.amax(np.abs(audio)))) + eps

    def __prepare_audio(self, audio_signal):
        """
        Crop audio and mix with white noise
        """

        # Adapt sample to windows size
        audio_length = audio_signal.shape[0]
        if(audio_length >= self.window_size):
            
            # If audio is bigger than window size use random crop: random shift
            left_bound = random.randint(0, audio_length - self.window_size)
            right_bound = left_bound + self.window_size
            audio_signal = audio_signal[left_bound:right_bound]
            
        else:
            # If the audio is smaller than the window size: pad original signal with 0z
            padding = self.window_size - audio_length
            bounds_sizes = np.random.multinomial(padding, np.ones(2)/2, size=1)[0]
            audio_signal = np.pad(
                audio_signal,
                (bounds_sizes[0], bounds_sizes[1]),
                'constant',
                constant_values=(0, 0)
                )

        # Add white noise
        if(self.white_noise):
            noise = au.generate_white_noise(len(audio_signal))
            noise = noise.astype(np.float32) / float(np.iinfo(noise.dtype).max)
            audio_signal += noise

        return audio_signal