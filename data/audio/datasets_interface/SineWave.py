import torch.utils.data as data
import numpy as np
import torch
from random import randint
import torch.nn.functional as F           # layers, activations and more
from torchaudio.transforms import MuLawEncoding


class SineWave(data.Dataset):
    """
    """
    FREQ = 440
    HZ = 8000
    CLASSES = 256

    def __init__(self, receptive_field):
        self.receptive_field = receptive_field

    def __getitem__(self, index):
        """ Returns a batch of random length created from a single song!
        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, int]: The output tuple (image, target) where target
            is index of the target class.
        """

        x = np.arange(8000*3)
        targets = np.sin(2 * np.pi * self.FREQ * (x/self.HZ))
        targets = torch.as_tensor(targets)
        targets = MuLawEncoding(quantization_channels=self.CLASSES)(targets)
        inputs = F.one_hot(targets, num_classes=self.CLASSES).permute(1, 0).float()

        return inputs[:, :-1], targets[self.receptive_field:].long()

    def __len__(self):
        return 1

if __name__ == '__main__':
    dta = SineWave()
    inp, tar = dta.__getitem__(0)
    print(inp.shape, tar.shape)