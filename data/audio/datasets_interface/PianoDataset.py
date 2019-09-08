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

    def __init__(self, root, batch_size, downsample=False, transform=None, target_transform=None, download=False,
                 transforms_on_creation=None, min_audio_length=2, max_audio_length=4):
        """

        :param root:
        :param downsample:
        :param transform:
        :param target_transform:
        :param download:
        :param transforms_on_creation:
        :param min_length: Minimum processed audio length in seconds
        :param max_length: Maximum processed audio length in seconds
        """
        if not self._check_exists(root):
           raise RuntimeError('Dataset not found.' +
                              ' You can use download=True to download it')

        self.root = root
        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.transforms_on_creation = transforms_on_creation
        self.chunk_size = 5
        self.num_samples = 330
        self.max_len = 0
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.cached_pt = 0
        self.batch_size = batch_size
        self.num_classes = 256
        self.sample_rate = 16000

        #if download:
        #    #self.download()
        #    self.process_audio()

    def __getitem__(self, index):
        """ Returns a batch of random length created from a single song!
        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, int]: The output tuple (image, target) where target
            is index of the target class.
        """

        data = torch.load(os.path.join(
            self.root, self.processed_folder, "piano_music_{:04d}.pt".format(self.cached_pt)))
        # cut out a random chunk of audio
        # TODO: check if max_audio_length is no longer than the audio itself
        segment_length = randint(self.min_audio_length * self.sample_rate, self.max_audio_length*self.sample_rate)

        rand_starts = (randint(0, len(data) - segment_length) for _ in range(self.batch_size))
        targets = torch.stack([data[start : start + segment_length] for start in rand_starts])
        audios = F.one_hot(targets, num_classes=self.num_classes)
        audios = audios.permute(0,2,1).float()

        #if self.transform is not None:
        #    audio = self.transform(audio)

        #target = None
        #if self.target_transform is not None:
        #    target = self.target_transform(audio)

        # inputs, targets
        return audios, targets[:, 1:]

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    dataset = PianoDataset('/media/jan//Data/datasets/PianoDataset', batch_size=1)
    loader = iter(DataLoader(dataset, batch_size=None, num_workers=1))
    for i, data in enumerate(loader):
        print(data[0].shape, data[1].shape)
        print(i)

    exit()

