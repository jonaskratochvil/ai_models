import os, glob
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, audio_folder, format='mp3'):

        self.data_index = glob.glob(audio_folder)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sound_name = os.path.join(self.audio_folder,
                                  self.data_index[idx])

        """
        sound = use torchsound for this..
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
        """
