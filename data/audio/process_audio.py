import os
import os.path
import torch, torchaudio
import json, subprocess
import errno

#from utils.utils import random_chunks

AUDIO_EXTENSIONS = [
    '.wav', '.mp3', '.flac', '.sph', '.ogg', '.opus',
    '.WAV', '.MP3', '.FLAC', '.SPH', '.OGG', '.OPUS',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def audio_files(dir):
    return[os.path.join(dir, audio) for audio in os.listdir(dir) if is_audio_file(audio)]


def check_dataset_exists(root, processed):
    return os.path.exists(os.path.join(root, processed, "info.txt"))


def read_audio(fp, downsample: int = 0):
    """Return first audio channel."""
    # FIXME: Test downsampling
    sig, sr = torchaudio.load(fp, filetype='mp3') # extract one channel only
    if downsample:
        sig = torchaudio.transforms.Resample(orig_freq=sr,
                                             new_freq=downsample, resampling_method='sinc_interpolation')(sig)
        sr = downsample

    return sig[0], sr


def scrap_from_url(url, file_type, root, processed_folder='processed', raw_folder='raw'):
    """Recursively scrap all files of `file_type` from `url`

    :param url: web adress
    :param file_type: for example 'mp3'
    :param root:
    :param processed_folder:
    :param raw_folder:
    :return:
    """

    if check_dataset_exists(root, processed_folder):
        return

    # download files
    try:
        os.makedirs(os.path.join(root, processed_folder))
        os.makedirs(os.path.join(root, raw_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    print('Downloading audio from ' + url)
    # FIXME: replace with python cross platform solution?
    cmd = "wget -c -A *.{} -r -l 1 -nd {} -P {}".format(file_type, url,
                                                        os.path.join(root, raw_folder))

    res = subprocess.run(cmd.split())
    if res.returncode != 0:
        print('Something went wrong during scrapping the page. Stderr:')
        print(res.stderr)


def process_audio(root, chunk_size=None, min_audio_length=None, max_audio_length=None, downsample=16000, transforms: list = None,
                  raw_folder='raw', processed_folder='processed'):
    """Process audio

    Apply each function from `transforms` to each audio
    Cut each audio into chunks of random size between (`min_audio_length`, `max_audio_length`).
    Save resulting audios into torch datafiles, where each datafile contains `chunk_size` origial audios
    (cut into smaller pieces).

    :param root: Root folder for data
    :param chunk_size: How many original audio files will be packed in one data file
    :param min_audio_length:
    :param max_audio_length:
    :param downsample: TODO
    :param transforms: A list of transform function to be applied to signals
    :param raw_folder: Name for raw data folder
    :param processed_folder: Name for processed data folder
    :return: None
    """

    if check_dataset_exists(raw_folder, processed_folder):
        print('Dataset already exists. Exiting...')
        return

    # process and save as torch files
    torchaudio.initialize_sox()
    print('Processing...')

    audios = audio_files(os.path.join(root, raw_folder))
    info = {'num_samples': len(audios),
            'samples':
                {audio:
                     {'sample_rate': 0,
                      'length': 0,
                      'dataset_id' : 0}
                 for audio in audios},
            }

    print("Found {} audio files".format(len(audios)))

    for i, audio in enumerate(audios):
        print('Processing ' + audio)
        sig, sample_rate = read_audio(audio, downsample=downsample)
        print(sig.shape, sample_rate)
        info['samples'][audio]['sample_rate'] = sample_rate
        info['samples'][audio]['length'] = len(sig)

        if transforms is not None:
            for transform in transforms:
                sig = transform(sig)

        torch.save(sig,
                   os.path.join(
                       root,
                       processed_folder,
                       "piano_music_{:04d}.pt".format(i))
                   )
        info['samples'][audio]['dataset_id'] = "piano_music_{:04d}.pt".format(i)

    with open(os.path.join(root, processed_folder, 'info.json'), 'w') as i:
        json.dump(info, i)
    torchaudio.shutdown_sox()
    print('Done!')


    """
    chunk_queue = []
    for audio in audios:
        print('Processing ' + audio)
        sig, sample_rate = read_audio(audio, downsample=downsample)
        info['sample_rate'] = sample_rate
        if transforms is not None:
            for transform in transforms:
                sig = transform(sig)

        chunk_queue += random_chunks(sig, min_audio_length * sample_rate, max_audio_length * sample_rate)
        for i in range(len(chunk_queue) // chunk_size):
            info['num_samples'] += chunk_size
            torch.save(chunk_queue[i*chunk_size: (i+1)*chunk_size],
                       os.path.join(
                           root,
                           processed_folder,
                           "piano_music_{:04d}.pt".format(i))
                       )
        chunk_queue = chunk_queue[(i+1)*chunk_size:]
    """



    """
    for n in range(len(audios) // chunk_size + 1):

        tensors, lengths = [], []
        st_idx = n * chunk_size
        end_idx = st_idx + chunk_size

        for i, f in enumerate(audios[st_idx:end_idx]):
            print('Processing ' + f)

            sig, sample_rate = read_audio(f, downsample=downsample)
            # TODO: adjust once sample rate normalisation is written
            info['sample_rate'] = sample_rate
            if transforms is not None:
                for transform in transforms:
                    sig = transform(sig)

            sig = random_chunks(sig, min_audio_length * sample_rate, max_audio_length * sample_rate)

            tensors += sig
            tmp = [len(s) for s in sig]
            info['max_len'] = max(info['max_len'], max(tmp))
            lengths += tmp

        # sort sigs/song_names: longest -> shortest
        tensors = [b for (a, b) in sorted(
            zip(lengths, tensors), key=lambda x: x[0], reverse=True)]
        data = tensors
        # Save as a list of tensors, each tensor is 1D
        torch.save(data,
                   os.path.join(
                       root,
                       processed_folder,
                       "piano_music_{:04d}.pt".format(n)
                   )
        )   

    info['num_samples'] = (n * chunk_size) + i + 1
    info['max_seconds'] = info['max_len'] // info['sample_rate'] + 1
    """

from torchaudio.transforms import MuLawEncoding

process_audio('/media/jan//Data/datasets/PianoDataset',
              transforms=[MuLawEncoding(quantization_channels=256)])
