from __future__ import absolute_import, division, print_function, unicode_literals
import torch.utils.data as data
from torchaudio.transforms import MuLawEncoding
import os
import os.path
import shutil
import errno
import torch
import torchaudio

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]
"""TODO:
1. Put used transforms and sampling rate into info file
2. Enable recreation of `processed` from raw data.
3. Cut audios into pieces of reasonable length (a couple seconds)
    - otherwise, feeding one audio might take hours
"""



def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)

def audio_files(dir):
    return[os.path.join(dir, audio) for audio in os.listdir(dir) if is_audio_file(audio)]

def read_audio(fp, downsample=False):
    """Return first audio channel."""
    if downsample:
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(fp)
        E.append_effect_to_chain("gain", ["-h"])
        E.append_effect_to_chain("channels", [1])
        E.append_effect_to_chain("rate", [16000])
        E.append_effect_to_chain("gain", ["-rh"])
        E.append_effect_to_chain("dither", ["-s"])
        sig, sr = E.sox_build_flow_effects()
    else:
        sig, sr = torchaudio.load(fp, filetype='mp3')
    sig = sig.contiguous() # FIXME: co to je?
    return sig[0], sr


class PianoDataset(data.Dataset):
    r"""Small Piano MIDI created dataset scrapped from `http://www.piano-midi.de/mp3.htm`

    Args:
        root (str): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        downsample (bool, optional): Whether to downsample the signal (Default: ``True``)
        transform (Callable, optional): A function/transform that takes in an raw audio
            and returns a transformed version. E.g, ``transforms.Spectrogram``. (Default: ``None``)
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. (Default: ``None``)
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. (Default: ``True``)
        dev_mode(bool, optional): If true, clean up is not performed on downloaded
            files.  Useful to keep raw audio and transcriptions. (Default: ``False``)
    """
    raw_folder = 'raw'
    processed_folder = 'processed'
    url = 'http://www.piano-midi.de/mp3.htm'
    dset_path = 'VCTK-Corpus'

    def __init__(self, root, downsample=False, transform=None, target_transform=None, download=False,
                 transforms_on_creation=None):
        self.root = root
        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.transforms_on_creation = transforms_on_creation
        self.data = []
        self.song_names = []
        self.chunk_size = 10
        self.num_samples = 0
        self.max_len = 0
        self.cached_pt = 0

        if download:
            #self.download()
            self.process_audio()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self._read_info()
        self.data, self.song_names = torch.load(os.path.join(
            self.root, self.processed_folder, "piano_music_{:04d}.pt".format(self.cached_pt)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            Tuple[torch.Tensor, int]: The output tuple (image, target) where target
            is index of the target class.
        """
        if self.cached_pt != index // self.chunk_size:
            self.cached_pt = index // self.chunk_size # set chunk pointer to the chunk where the index resides
            self.data, self.song_names = torch.load(os.path.join(
                self.root, self.processed_folder, "piano_music_{:04d}.pt".format(self.cached_pt)))
        index = index % self.chunk_size
        audio, song_names = self.data[index], self.song_names[index]

        if self.transform is not None:
            audio = self.transform(audio)

        target = None
        if self.target_transform is not None:
            target = self.target_transform(audio)

        return audio, target, song_names

    def __len__(self):
        return self.num_samples

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "info.txt"))

    def _write_info(self, num_items):
        info_path = os.path.join(
            self.root, self.processed_folder, "info.txt")
        with open(info_path, "w") as f:
            f.write("num_samples,{}\n".format(num_items))
            f.write("max_len,{}\n".format(self.max_len))

    def _read_info(self):
        info_path = os.path.join(
            self.root, self.processed_folder, "info.txt")
        with open(info_path, "r") as f:
            self.num_samples = int(f.readline().split(",")[1])
            self.max_len = int(f.readline().split(",")[1])

    def download(self):
        """Download the data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        import subprocess

        raw_abs_dir = os.path.join(self.root, self.raw_folder)
        processed_abs_dir = os.path.join(self.root, self.processed_folder)

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading audio from ' + self.url)
        # FIXME: replace with python cross platform solution
        cmd = "wget -c -A *.mp3 -r -l 1 -nd {} -P {}".format(self.url,
                                                               os.path.join(self.root, self.raw_folder)
                                                               )
        res = subprocess.run(cmd.split())


    def process_audio(self):

        if self._check_exists():
            return

        # process and save as torch files
        torchaudio.initialize_sox()
        print('Processing...')

        audios = audio_files(os.path.join(self.root, self.raw_folder))
        self.max_len = 0
        print("Found {} audio files".format(len(audios)))

        for n in range(len(audios) // self.chunk_size + 1):
            tensors = []
            song_names = []
            lengths = []
            st_idx = n * self.chunk_size
            end_idx = st_idx + self.chunk_size
            for i, f in enumerate(audios[st_idx:end_idx]):
                sig = read_audio(f, downsample=self.downsample)[0]

                if self.transforms_on_creation is not None:
                    for transform in self.transforms_on_creation:
                        sig = transform(sig)

                tensors.append(sig)
                lengths.append(sig.size(1))
                song_names.append(f)
                self.max_len = max(self.max_len, sig.size(1))

            # sort sigs/song_names: longest -> shortest
            tensors, song_names = zip(*[(b, c) for (a, b, c) in sorted(
                zip(lengths, tensors, song_names), key=lambda x: x[0], reverse=True)])
            data = (tensors, song_names)
            torch.save(data,
                       os.path.join(
                           self.root,
                           self.processed_folder,
                           "piano_music_{:04d}.pt".format(n)
                       )
            )
        self._write_info((n * self.chunk_size) + i + 1)
        torchaudio.shutdown_sox()
        print('Done!')

if __name__ == '__main__':

    import torch.nn.functional as F

    dataset = PianoDataset('/Users/vainerj/PianoDataset', download=False,
                           transforms_on_creation=[MuLawEncoding(quantization_channels=256)])
    audio, _, _ = dataset[1]
    # takes really long time for even 2 minutes of audio
    # -> preprocess during dataset creation? Use sparse tensors?
    oh = F.one_hot(audio[0], num_classes=256)
    print(oh)
