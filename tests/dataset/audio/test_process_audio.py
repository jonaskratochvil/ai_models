import pytest, os, sys, json
import torch

# import project root to PYTHONPATH
abs_dir = os.path.dirname(os.path.abspath(__file__)) # abs adress of this file
root_folder = abs_dir.split('tests')[0]
sys.path.append(root_folder)

from dataset.audio.process_audio import process_audio


def test_process_audio():

    root = '/Users/vainerj/PianoDataset'
    min_audio_length = 5
    max_audio_length = 5
    chunk_size = 5

    # process_audio(root=root,
    #               chunk_size=chunk_size,
    #               min_audio_length=min_audio_length,  # seconds
    #               max_audio_length=max_audio_length,  # seconds
    #               )

    with open(os.path.join(root, 'processed', 'info.json')) as i:
        info = json.loads(i)

    x = torch.load(os.path.join(
                root, 'processed', "piano_music_0001.pt"))
    #assert len(x) == 5  # the chunk size - will fail for now
    for signal in x:
        assert min_audio_length * info['sample_rate'] <= len(signal) <= max_audio_length * info['sample_rate']

