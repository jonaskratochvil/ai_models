import torch.utils.data as data
import os
import os.path
import torch
from random import randint
import torch.nn.functional as F           # layers, activations and more


class PianoDataset(data.Dataset):
    r"""Small Piano MIDI created data scrapped from `http://www.piano-midi.de/mp3.htm`

    Args:
        root (str): Root directory of data where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        downsample (bool, optional): Whether to downsample the signal (Default: ``True``)
        transform (Callable, optional): A function/transform that takes in an raw audio
            and returns a transformed version. E.g, ``transforms.Spectrogram``. (Default: ``None``)
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. (Default: ``None``)
        download (bool, optional): If true, downloads the data from the internet and
            puts it in root directory. If data is already downloaded, it is not
            downloaded again. (Default: ``True``)
        dev_mode(bool, optional): If true, clean up is not performed on downloaded
            files.  Useful to keep raw audio and transcriptions. (Default: ``False``)
    """
    raw_folder = 'raw'
    processed_folder = 'processed'
    url = 'http://www.piano-midi.de/mp3.htm'

    def _check_exists(self, root):
        return os.path.exists(os.path.join(root, self.processed_folder, "info.json"))

    def __init__(self, root, batch_size, min_length, num_targets, num_classes):
        """
        """
        if not self._check_exists(root):
           raise RuntimeError('Dataset not found.' +
                              ' You can use download=True to download it')

        self.root = root
        self.batch_size = batch_size
        self.min_lenght = min_length
        self.num_targets = num_targets
        self.num_classes = num_classes
        self.num_samples = 300 # FIXME

    def __getitem__(self, index):
        """ Returns a batch of random length created from a single song!
        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, int]: The output tuple (image, target) where target
            is index of the target class.
        """

        data = torch.load(os.path.join(
            self.root, self.processed_folder, "piano_music_{:04d}.pt".format(index)))
        rand_start = randint(0, len(data) - (self.min_lenght + self.num_targets))
        inputs = torch.as_tensor(data[rand_start : rand_start + self.min_lenght + self.num_targets - 1])
        targets = torch.as_tensor(data[rand_start + self.min_lenght : rand_start + self.min_lenght + self.num_targets])
        inputs =  F.one_hot(inputs, num_classes=self.num_classes).permute(1, 0).float()

        return inputs, targets

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    dataset = PianoDataset('/media/jan//Data/datasets/PianoDataset', batch_size=10, min_length=100, num_targets=10, num_classes=256,)
    loader = iter(DataLoader(dataset, batch_size=10, num_workers=0))
    for i, data in enumerate(loader):
        print(data[0].shape, data[1].shape)

    exit()

