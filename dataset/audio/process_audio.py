import os
import os.path
import torch, torchaudio
import json

from utils.utils import random_chunks

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


def read_audio(fp, downsample=False):
    """Return first audio channel."""
    # FIXME: Test downsampling
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


def process_audio(root, chunk_size, min_audio_length, max_audio_length, downsample=False, transforms: list = None,
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

    # TODO: normalize sample rate across samples - use librosa for that

    if check_dataset_exists(raw_folder, processed_folder):
        print('Dataset already exists. Exiting...')
        return

    # process and save as torch files
    torchaudio.initialize_sox()
    print('Processing...')

    audios = audio_files(os.path.join(root, raw_folder))
    info = {'max_len' : 0,
            'sample_rate' : None}

    print("Found {} audio files".format(len(audios)))

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

            print(sig.shape)
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

    with open(os.path.join(root, processed_folder, 'info.json'), 'w') as i:
        json.dump(info, i)
    torchaudio.shutdown_sox()
    print('Done!')


